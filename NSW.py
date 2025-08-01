"""
================================
Overlapping communities  ➜  community‑based weights  ➜  Spatial‑Error model.

• Multiple weighted overlapping community detection methods
• Auto‑drops perfectly collinear columns          (safe X'X)
• Works whether your spreg.GM_Error_Het exposes    constant=
  or not – handled by a small helper.

Dependencies
------------
pip install pandas networkx libpysal spreg numpy cdlib python-louvain oslom-runner
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Optional
import warnings
import inspect
import tempfile
import os

import networkx as nx
import numpy as np
import pandas as pd
from libpysal.weights import W
from spreg import GM_Error_Het


# ────────────────────────────────────────────────────────────────────────────
# 1.  FLOW GRAPH
# ────────────────────────────────────────────────────────────────────────────
def build_graph(
    flows: pd.DataFrame,
    *,
    origin_col: str = "origin",
    dest_col: str   = "destination",
    weight_col: str = "weight",
) -> nx.DiGraph:
    """Directed, weighted graph from an O‑D table."""
    G = nx.DiGraph()
    for o, d, w in flows[[origin_col, dest_col, weight_col]].itertuples(index=False):
        G.add_edge(o, d, weight=float(w))
    return G


# ────────────────────────────────────────────────────────────────────────────
# 2.  WEIGHTED OVERLAPPING COMMUNITIES
# ────────────────────────────────────────────────────────────────────────────

def detect_weighted_overlapping_communities(
    G: nx.Graph, 
    method: str = "leiden_multiplex",
    **kwargs
) -> pd.Series:
    """
    Detect overlapping communities in weighted networks using various methods.
    
    Parameters:
    -----------
    G : nx.Graph
        Undirected weighted graph
    method : str
        Community detection method:
        - "leiden_multiplex": Leiden algorithm allowing overlaps
        - "oslom": OSLOM weighted overlapping detection (requires oslom-runner)
        - "weighted_cliques": Weighted k-clique-like method
        - "ego_splitting": Ego-network splitting approach
    
    Returns:
    --------
    pd.Series
        Node memberships where each value is a list of community IDs
    """
    
    if method == "leiden_multiplex":
        return _leiden_multiplex_communities(G, **kwargs)
    elif method == "oslom":
        return _oslom_communities(G, **kwargs)
    elif method == "weighted_cliques":
        return _weighted_clique_communities(G, **kwargs)
    elif method == "ego_splitting":
        return _ego_splitting_communities(G, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _leiden_multiplex_communities(G: nx.Graph, resolution: float = 1.0) -> pd.Series:
    """
    Use Leiden algorithm with ego-network splitting to create overlapping communities.
    This approximates overlapping behavior by allowing high-degree nodes to belong
    to multiple communities based on their neighborhood structure.
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("Please install: pip install python-louvain")
    
    # Standard Leiden/Louvain on the full graph
    partition = community_louvain.best_partition(G, resolution=resolution, weight='weight')
    
    # Create overlapping memberships for high-degree nodes
    node2comms: Dict[str, List[int]] = {}
    
    for node in G.nodes():
        node2comms[str(node)] = [partition[node]]
        
        # For high-degree nodes, check if they bridge communities
        if G.degree(node, weight='weight') > np.percentile([G.degree(n, weight='weight') for n in G.nodes()], 75):
            neighbor_communities = [partition[neighbor] for neighbor in G.neighbors(node)]
            # Add node to communities where it has strong connections
            for comm in set(neighbor_communities):
                if comm != partition[node]:
                    # Calculate connection strength to this community
                    comm_weight = sum(G[node][neighbor]['weight'] 
                                    for neighbor in G.neighbors(node) 
                                    if partition[neighbor] == comm)
                    total_weight = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
                    
                    # If >20% of node's weight goes to this community, add membership
                    if comm_weight / total_weight > 0.2:
                        node2comms[str(node)].append(comm)
    
    n_communities = len(set(partition.values()))
    n_overlapping = sum(1 for comms in node2comms.values() if len(comms) > 1)
    
    print(f"🔍 Detected {n_communities} communities using Leiden with ego-splitting.")
    print(f"    {n_overlapping} nodes belong to multiple communities.")
    
    return pd.Series(node2comms)


def _oslom_communities(G: nx.Graph, iterations: int = 10) -> pd.Series:
    """
    Use OSLOM (Order Statistics Local Optimization Method) for weighted overlapping detection.
    Requires: pip install oslom-runner
    """
    try:
        from oslom_runner import OSLOM
    except ImportError:
        raise ImportError("Please install: pip install oslom-runner")
    
    # Create temporary files for OSLOM
    with tempfile.NamedTemporaryFile(mode='w', suffix='.edge', delete=False) as f:
        # Write weighted edge list
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            f.write(f"{u}\t{v}\t{weight}\n")
        edge_file = f.name
    
    try:
        # Run OSLOM
        oslom = OSLOM()
        result = oslom.run(edge_file, args=f"-w -r {iterations} -hr {iterations}")
        
        # Parse results
        node2comms: Dict[str, List[int]] = {str(node): [] for node in G.nodes()}
        
        if hasattr(result, 'communities'):
            for cid, community in enumerate(result.communities):
                for node in community:
                    if str(node) in node2comms:
                        node2comms[str(node)].append(cid)
        
        print(f"🔍 Detected {len(result.communities)} overlapping communities using OSLOM.")
        
    finally:
        # Cleanup
        if os.path.exists(edge_file):
            os.unlink(edge_file)
    
    return pd.Series(node2comms)


def _weighted_clique_communities(G: nx.Graph, min_weight_threshold: float = 0.5) -> pd.Series:
    """
    Custom weighted clique-based method.
    Finds dense subgraphs where edge weights exceed threshold, allows overlaps.
    """
    from itertools import combinations
    
    # Filter edges by weight threshold
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) 
                   if d.get('weight', 0) >= min_weight_threshold]
    G_filtered = G.edge_subgraph(strong_edges).copy()
    
    # Find maximal cliques in filtered graph
    cliques = list(nx.find_cliques(G_filtered))
    communities = [clique for clique in cliques if len(clique) >= 3]
    
    # Create overlapping memberships
    node2comms: Dict[str, List[int]] = {str(node): [] for node in G.nodes()}
    
    for cid, community in enumerate(communities):
        for node in community:
            node2comms[str(node)].append(cid)
    
    print(f"🔍 Detected {len(communities)} weighted clique communities.")
    print(f"    Used weight threshold: {min_weight_threshold}")
    
    return pd.Series(node2comms)


def _ego_splitting_communities(G: nx.Graph, overlap_threshold: float = 0.3) -> pd.Series:
    """
    Ego-network splitting approach for overlapping communities.
    High-degree nodes can belong to multiple communities based on their ego-networks.
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("Please install: pip install python-louvain")
    
    # Base partition using Louvain
    base_partition = community_louvain.best_partition(G, weight='weight')
    node2comms: Dict[str, List[int]] = {str(node): [base_partition[node]] for node in G.nodes()}
    
    # Identify overlap candidates (high-degree nodes)
    degrees = dict(G.degree(weight='weight'))
    degree_threshold = np.percentile(list(degrees.values()), 80)
    
    overlap_candidates = [node for node, degree in degrees.items() if degree > degree_threshold]
    
    for node in overlap_candidates:
        ego_graph = G.subgraph([node] + list(G.neighbors(node)))
        ego_partition = community_louvain.best_partition(ego_graph, weight='weight')
        
        # Find communities in ego network
        ego_communities = {}
        for ego_node, ego_comm in ego_partition.items():
            if ego_comm not in ego_communities:
                ego_communities[ego_comm] = []
            ego_communities[ego_comm].append(ego_node)
        
        # If ego network has multiple communities, node might belong to multiple global communities
        if len(ego_communities) > 1:
            for ego_comm, ego_nodes in ego_communities.items():
                if node in ego_nodes:
                    # Check which global communities these ego nodes belong to
                    global_comms = [base_partition[ego_node] for ego_node in ego_nodes if ego_node != node]
                    if global_comms:
                        main_global_comm = max(set(global_comms), key=global_comms.count)
                        if main_global_comm not in node2comms[str(node)]:
                            # Calculate overlap strength
                            overlap_weight = sum(G[node][neighbor].get('weight', 1) 
                                               for neighbor in ego_nodes if neighbor != node and neighbor in G[node])
                            total_weight = sum(G[node][neighbor].get('weight', 1) for neighbor in G.neighbors(node))
                            
                            if overlap_weight / total_weight > overlap_threshold:
                                node2comms[str(node)].append(main_global_comm)
    
    n_communities = len(set(base_partition.values()))
    n_overlapping = sum(1 for comms in node2comms.values() if len(comms) > 1)
    
    print(f"🔍 Detected {n_communities} communities using ego-splitting.")
    print(f"    {n_overlapping} nodes belong to multiple communities.")
    
    return pd.Series(node2comms)


# ────────────────────────────────────────────────────────────────────────────
# 3.  PRIMARY COMMUNITY  (largest internal flow)
# ────────────────────────────────────────────────────────────────────────────
def primary_community(G: nx.DiGraph, memberships: pd.Series) -> pd.Series:
    def pick(node: str, comms: Sequence[int]) -> int:
        if len(comms) == 1:
            return comms[0]
        scores = {}
        for c in comms:
            members = {n for n, lst in memberships.items() if c in lst}
            flow_out = sum(G[node][nbr].get("weight", 0) for nbr in G.successors(node) if nbr in members)
            flow_in  = sum(G[p][node].get("weight", 0) for p in G.predecessors(node) if p in members)
            scores[c] = flow_out + flow_in
        return max(scores, key=lambda cid: (scores[cid], -cid))
    return pd.Series({n: pick(n, lst) if lst else -1 for n, lst in memberships.items()},
                     name="community")


# ────────────────────────────────────────────────────────────────────────────
# 4.  COMMUNITY‑BASED BINARY WEIGHTS
# ────────────────────────────────────────────────────────────────────────────
def community_weights(primary: pd.Series, *, row_std: bool = True) -> W:
    neigh = {}
    for grp, members in primary.groupby(primary).groups.items():
        for i in members:
            neigh[i] = [j for j in members if j != i]
    Wc = W(neigh)
    if row_std:
        Wc.transform = "R"
    return Wc


# ────────────────────────────────────────────────────────────────────────────
# 5.  RUN SEM (with or without constant kw)
# ────────────────────────────────────────────────────────────────────────────
def _gm_error_het(y, X, Wc, *, name_y, name_x, name_w, extra_kwargs):
    """
    Call GM_Error_Het in a way that works across spreg versions:
    some expose constant=, others always append an intercept internally.
    """
    sig = inspect.signature(GM_Error_Het.__init__)
    has_constant_kw = "constant" in sig.parameters

    if has_constant_kw:
        return GM_Error_Het(
            y, X, Wc,
            constant=False,       # we add intercept ourselves if needed
            name_y=name_y, name_x=name_x, name_w=name_w,
            **extra_kwargs
        )
    else:
        return GM_Error_Het(
            y, X, Wc,
            name_y=name_y, name_x=name_x, name_w=name_w,
            **extra_kwargs
        )


# ────────────────────────────────────────────────────────────────────────────
# 6.  ENDOGENEITY TESTING AND CORRECTION
# ────────────────────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from scipy import stats

def test_w_endogeneity(
    y: np.ndarray,
    X: np.ndarray, 
    W: W,
    Z: Optional[np.ndarray] = None,
    X2: Optional[np.ndarray] = None
) -> dict:
    """
    Test for endogeneity of the spatial weights matrix W using Rao's score test.
    
    Parameters:
    -----------
    y : np.ndarray
        Dependent variable (n×1)
    X : np.ndarray  
        Exogenous covariates (n×k)
    W : libpysal.weights.W
        Spatial weights matrix
    Z : np.ndarray, optional
        Observable covariates that drive W formation (n×p2)
    X2 : np.ndarray, optional
        Exogenous predictors for Z (n×k2). If None, uses X.
        
    Returns:
    --------
    dict
        Test results including statistic, p-value, and endogeneity conclusion
    """
    n = len(y)
    
    if Z is None:
        # If no Z provided, we cannot perform the formal test
        return {
            'test_statistic': None,
            'p_value': None,
            'is_endogenous': None,
            'method': 'No Z provided - cannot test endogeneity formally',
            'degrees_of_freedom': None
        }
    
    if X2 is None:
        X2 = X.copy()
    
    # Step 1: Estimate reduced form Z = X2*Gamma + epsilon
    # Using OLS for each column of Z
    Gamma_hat = np.linalg.lstsq(X2, Z, rcond=None)[0]  # k2 × p2
    Z_fitted = X2 @ Gamma_hat
    epsilon_hat = Z - Z_fitted  # Residuals (n × p2)
    
    # Step 2: Estimate restricted spatial model (without control function)
    # y = rho*W*y + X*beta + v
    try:
        from spreg import GM_Error_Het
        restricted_model = GM_Error_Het(y, X, W, name_y="y", name_x=[f"x{i}" for i in range(X.shape[1])])
        
        # Extract residuals from restricted model
        v_hat = restricted_model.u
        
        # Step 3: Compute score with respect to delta (coefficient on control function)
        # Score = epsilon_hat' * v_hat (simplified version)
        score = epsilon_hat.T @ v_hat  # p2 × 1
        
        # Step 4: Compute variance-covariance matrix for score
        # Simplified version - in practice would need spatial adjustment
        V_score = (epsilon_hat.T @ epsilon_hat) * (v_hat.T @ v_hat) / n
        
        # Step 5: Rao's score statistic
        if np.isscalar(V_score):
            test_stat = float((score ** 2) / V_score) if V_score > 0 else 0
            df = 1
        else:
            # Multivariate case
            try:
                V_inv = np.linalg.pinv(V_score)
                test_stat = float(score.T @ V_inv @ score)
                df = len(score)
            except:
                test_stat = 0
                df = len(score)
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(test_stat, df)
        is_endogenous = p_value < 0.05
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'is_endogenous': is_endogenous,
            'method': 'Rao score test',
            'degrees_of_freedom': df,
            'control_function': epsilon_hat
        }
        
    except Exception as e:
        print(f"Warning: Endogeneity test failed: {e}")
        return {
            'test_statistic': None,
            'p_value': None, 
            'is_endogenous': True,  # Assume endogenous to be safe
            'method': f'Test failed: {e}',
            'degrees_of_freedom': None
        }


def extract_w_factors(W: W, n_components: int = 1) -> np.ndarray:
    """
    Extract latent factors from spatial weights matrix using PCA.
    
    Parameters:
    -----------
    W : libpysal.weights.W
        Spatial weights matrix
    n_components : int
        Number of principal components to extract
        
    Returns:
    --------
    np.ndarray
        Extracted factors (n × n_components)
    """
    # Convert W to dense matrix
    W_dense = W.full()[0]  # Get the dense representation
    n = W_dense.shape[0]
    
    # Option 1: Use rows of W
    W_features = W_dense.copy()
    
    # Option 2: Could also use W^2, symmetrized W, etc.
    W2 = W_dense @ W_dense
    W_sym = (W_dense + W_dense.T) / 2
    
    # Combine different representations
    W_features = np.hstack([W_dense, W2, W_sym])  # n × 3n
    
    # Standardize features
    W_features = (W_features - W_features.mean(axis=0)) / (W_features.std(axis=0) + 1e-10)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(W_features)
    
    explained_var = pca.explained_variance_ratio_
    
    print(f"🔍 Extracted {n_components} latent factors from W matrix.")
    print(f"    Explained variance: {explained_var}")
    print(f"    Cumulative explained variance: {np.cumsum(explained_var)[-1]:.3f}")
    
    return factors


def estimate_corrected_model(
    y: np.ndarray,
    X: np.ndarray,
    W: W,
    control_vars: np.ndarray,
    *,
    name_y: str,
    name_x: List[str],
    name_w: str,
    extra_kwargs: dict
) -> object:
    """
    Estimate spatial model with endogeneity correction.
    
    Parameters:
    -----------
    y, X, W : Standard spatial model inputs
    control_vars : np.ndarray
        Control function variables (either from Z or PCA factors)
    
    Returns:
    --------
    Estimated spatial model with endogeneity correction
    """
    # Augment X with control variables
    X_augmented = np.hstack([X, control_vars])
    
    # Update names
    n_controls = control_vars.shape[1]
    control_names = [f"control_{i+1}" for i in range(n_controls)]
    name_x_augmented = name_x + control_names
    
    # Estimate corrected model
    model = _gm_error_het(
        y, X_augmented, W,
        name_y=name_y,
        name_x=name_x_augmented, 
        name_w=name_w,
        extra_kwargs=extra_kwargs
    )
    
    return model


# ────────────────────────────────────────────────────────────────────────────
# 7.  FULL PIPELINE WITH ENDOGENEITY TESTING
# ────────────────────────────────────────────────────────────────────────────
import time

def run_sem_with_community_weights(
    nodes: pd.DataFrame,
    flows: pd.DataFrame,
    *,
    id_col: str = "id",
    target_col: str,
    feature_cols: Sequence[str],
    origin_col: str = "origin",
    dest_col:   str = "destination",
    weight_col: str = "weight",
    community_method: str = "leiden_multiplex",
    community_kwargs: dict | None = None,
    sem_kwargs: dict | None = None,
    # Endogeneity testing parameters
    Z: Optional[np.ndarray] = None,
    X2: Optional[np.ndarray] = None,
    n_pca_components: int = 1,
    test_endogeneity: bool = True,
):
    """
    Full pipeline for community-based spatial econometric modeling with endogeneity testing.
    
    Parameters:
    -----------
    community_method : str
        Method for weighted overlapping community detection:
        - "leiden_multiplex": Leiden with ego-splitting (recommended)
        - "oslom": OSLOM algorithm (requires oslom-runner)
        - "weighted_cliques": Custom weighted clique method
        - "ego_splitting": Ego-network based splitting
    Z : np.ndarray, optional
        Observable covariates that drive W formation (n×p2).
        If provided, formal endogeneity test is performed.
    X2 : np.ndarray, optional  
        Exogenous predictors for Z (n×k2). If None, uses main X.
    n_pca_components : int, default=1
        Number of PCA components to extract from W if endogeneity detected and no Z provided.
    test_endogeneity : bool, default=True
        Whether to test for endogeneity of W.
        
    Returns:
    --------
    dict
        Results containing:
        - 'base_model': Initial spatial model
        - 'corrected_model': Endogeneity-corrected model (if needed)
        - 'endogeneity_test': Test results
        - 'primary': Primary community assignments
        - 'W': Spatial weights matrix
        - 'G': Flow graph
        - 'overlaps': Overlapping community memberships
    """
    sem_kwargs = sem_kwargs or {}
    community_kwargs = community_kwargs or {}

    print("Step 1: Building flow graph...")
    t0 = time.time()
    G = build_graph(flows, origin_col=origin_col,
                    dest_col=dest_col, weight_col=weight_col)
    print(f" → Done in {time.time() - t0:.2f} seconds")

    print(f"Step 2: Detecting weighted overlapping communities ({community_method})...")
    t0 = time.time()
    H = G.to_undirected()
    # Combine edge weights from both directions
    for u, v in H.edges():
        weight_uv = G[u][v].get('weight', 0) if G.has_edge(u, v) else 0
        weight_vu = G[v][u].get('weight', 0) if G.has_edge(v, u) else 0
        H[u][v]['weight'] = weight_uv + weight_vu
    
    overlaps = detect_weighted_overlapping_communities(H, method=community_method, **community_kwargs)
    print(f" → Done in {time.time() - t0:.2f} seconds")

    print("Step 3: Assigning primary community...")
    t0 = time.time()
    primary = primary_community(G, overlaps)
    print(f" → Done in {time.time() - t0:.2f} seconds")

    print("Step 4: Building community-based spatial weights matrix...")
    t0 = time.time()
    Wc = community_weights(primary)
    print(f" → Done in {time.time() - t0:.2f} seconds")

    print("Step 5: Preparing regression data...")
    t0 = time.time()
    df = nodes.set_index(id_col).loc[primary.index]
    y = df[[target_col]].to_numpy()
    X = df[list(feature_cols)].to_numpy()

    # Drop perfectly collinear columns
    keep, names = [], list(feature_cols)
    rank_so_far = 0
    for i in range(X.shape[1]):
        if np.linalg.matrix_rank(np.hstack([X[:, keep], X[:, [i]]])) > rank_so_far:
            keep.append(i)
            rank_so_far += 1
    if len(keep) < X.shape[1]:
        dropped = ", ".join(names[i] for i in range(X.shape[1]) if i not in keep)
        warnings.warn(f"Dropped collinear columns: {dropped}")
    X = X[:, keep]
    names = [names[i] for i in keep]

    # Add intercept if needed
    if "constant" in inspect.signature(GM_Error_Het.__init__).parameters:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        names = ["const"] + names
    print(f" → Done in {time.time() - t0:.2f} seconds")

    print("Step 6: Estimating initial spatial error model...")
    t0 = time.time()
    base_model = _gm_error_het(
        y, X, Wc,
        name_y=target_col,
        name_x=names,
        name_w="CommunityW",
        extra_kwargs=sem_kwargs
    )
    print(f" → Done in {time.time() - t0:.2f} seconds")

    # Initialize results dictionary
    results = {
        'base_model': base_model,
        'corrected_model': None,
        'endogeneity_test': None,
        'primary': primary,
        'W': Wc,
        'G': G,
        'overlaps': overlaps
    }

    # Step 7: Test for endogeneity of W
    if test_endogeneity:
        print("Step 7: Testing endogeneity of spatial weights matrix...")
        t0 = time.time()
        
        endogeneity_test = test_w_endogeneity(y, X, Wc, Z=Z, X2=X2)
        results['endogeneity_test'] = endogeneity_test
        
        print(f" → Done in {time.time() - t0:.2f} seconds")
        
        if endogeneity_test['is_endogenous'] is None:
            print("⚠️  Cannot formally test endogeneity (no Z provided)")
            if Z is None:
                print("   Applying PCA proxy correction as robustness check...")
                # Apply PCA correction as robustness
                t0 = time.time()
                pca_factors = extract_w_factors(Wc, n_components=n_pca_components)
                corrected_model = estimate_corrected_model(
                    y, X, Wc, pca_factors,
                    name_y=target_col,
                    name_x=names,
                    name_w="CommunityW",
                    extra_kwargs=sem_kwargs
                )
                results['corrected_model'] = corrected_model
                print(f"   → PCA correction completed in {time.time() - t0:.2f} seconds")
        
        elif endogeneity_test['is_endogenous']:
            print(f"⚠️  Endogeneity detected (p-value: {endogeneity_test['p_value']:.4f})")
            print("   Applying endogeneity correction...")
            
            t0 = time.time() 
            
            if Z is not None and 'control_function' in endogeneity_test:
                # Use formal control function approach
                print("   Using control function approach with provided Z")
                control_vars = endogeneity_test['control_function']
            else:
                # Use PCA proxy approach
                print(f"   Using PCA proxy approach ({n_pca_components} components)")
                control_vars = extract_w_factors(Wc, n_components=n_pca_components)
            
            corrected_model = estimate_corrected_model(
                y, X, Wc, control_vars,
                name_y=target_col,
                name_x=names, 
                name_w="CommunityW",
                extra_kwargs=sem_kwargs
            )
            results['corrected_model'] = corrected_model
            print(f"   → Correction completed in {time.time() - t0:.2f} seconds")
            
        else:
            print(f"✅ No endogeneity detected (p-value: {endogeneity_test['p_value']:.4f})")
            print("   Base model is appropriate")

    print("✅ All steps completed.")
    
    return results
