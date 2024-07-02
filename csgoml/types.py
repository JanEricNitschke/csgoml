"""This module contains the type definitions for the parsed json structure."""

from typing import Literal, TypedDict, final

IDNumberDict = dict[Literal["ct", "t"], dict[str, Literal["1", "2", "3", "4", "5"]]]

DictInitialized = dict[Literal["ct", "t"], bool]

SideSelection = Literal["CT", "T", "BOTH"]

CoordinateTypes = Literal["area", "token", "position"]


class AllowedForbidden(TypedDict):
    """Holds information about which values are allowed or forbidden."""

    Allowed: list[str]
    Forbidden: list[str]


class EquipSpecification(TypedDict):
    """Holds information about equipment specifications."""

    Kill: list[str]
    CT: AllowedForbidden
    T: AllowedForbidden


class PositionSpecification(TypedDict):
    """Holds information about which positions are of interest."""

    CT: list[str]
    T: list[str]


class WeaponsClassesSwitch(TypedDict):
    """Holds information about whether to filter by weapons or classes."""

    CT: Literal["weapons", "classes"]
    T: Literal["weapons", "classes"]
    Kill: Literal["weapons", "classes"]


class TimeSpecification(TypedDict):
    """Holds information about the time the fights should have occurred in."""

    start: int
    end: int


class FightSpecification(TypedDict):
    """Holds information about fights to parse."""

    map_name: str
    weapons: EquipSpecification
    classes: EquipSpecification
    positions: PositionSpecification
    use_weapons_classes: WeaponsClassesSwitch
    times: TimeSpecification


class QueryResult(TypedDict):
    """Holds information about result for querying a specific fight situation."""

    situations_found: int
    ct_win_percentage: tuple[float, float, float]


class TickInformation(TypedDict):
    """Holds information about tickrate and freezetimeendtick."""

    tickRate: int
    freezeTimeEndTick: int


class RoundPositions(TypedDict):
    """Holds information about player positions during a round."""

    MatchID: str
    MapName: str
    Round: int
    Winner: Literal[0, 1]
    Tick: list[int]
    token: list[str]
    interpolated: list[Literal[0, 1]]
    CTtoken: list[str]
    CTPlayer1Alive: list[Literal[0, 1]]
    CTPlayer1Name: list[str]
    CTPlayer1x: list[float]
    CTPlayer1y: list[float]
    CTPlayer1z: list[float]
    CTPlayer1Area: list[int]
    CTPlayer2Alive: list[Literal[0, 1]]
    CTPlayer2Name: list[str]
    CTPlayer2x: list[float]
    CTPlayer2y: list[float]
    CTPlayer2z: list[float]
    CTPlayer2Area: list[int]
    CTPlayer3Alive: list[Literal[0, 1]]
    CTPlayer3Name: list[str]
    CTPlayer3x: list[float]
    CTPlayer3y: list[float]
    CTPlayer3z: list[float]
    CTPlayer3Area: list[int]
    CTPlayer4Alive: list[Literal[0, 1]]
    CTPlayer4Name: list[str]
    CTPlayer4x: list[float]
    CTPlayer4y: list[float]
    CTPlayer4z: list[float]
    CTPlayer4Area: list[int]
    CTPlayer5Alive: list[Literal[0, 1]]
    CTPlayer5Name: list[str]
    CTPlayer5x: list[float]
    CTPlayer5y: list[float]
    CTPlayer5z: list[float]
    CTPlayer5Area: list[int]
    Ttoken: list[str]
    TPlayer1Alive: list[Literal[0, 1]]
    TPlayer1Name: list[str]
    TPlayer1x: list[float]
    TPlayer1y: list[float]
    TPlayer1z: list[float]
    TPlayer1Area: list[int]
    TPlayer2Alive: list[Literal[0, 1]]
    TPlayer2Name: list[str]
    TPlayer2x: list[float]
    TPlayer2y: list[float]
    TPlayer2z: list[float]
    TPlayer2Area: list[int]
    TPlayer3Alive: list[Literal[0, 1]]
    TPlayer3Name: list[str]
    TPlayer3x: list[float]
    TPlayer3y: list[float]
    TPlayer3z: list[float]
    TPlayer3Area: list[int]
    TPlayer4Alive: list[Literal[0, 1]]
    TPlayer4Name: list[str]
    TPlayer4x: list[float]
    TPlayer4y: list[float]
    TPlayer4z: list[float]
    TPlayer4Area: list[int]
    TPlayer5Alive: list[Literal[0, 1]]
    TPlayer5Name: list[str]
    TPlayer5x: list[float]
    TPlayer5y: list[float]
    TPlayer5z: list[float]
    TPlayer5Area: list[int]


class PositionDatasetJSON(TypedDict):
    """Holds information about the rounds of a game."""

    MatchID: list[str]
    MapName: list[str]
    Round: list[int]
    Winner: list[Literal[0, 1]]
    position_df: list[RoundPositions]


@final
class TrajectoryConfig(TypedDict):
    """Holds information about trajectories to analyze."""

    coordinate_type_for_distance: CoordinateTypes
    n_rounds: int
    time: int
    side: SideSelection
    dtw: bool


@final
class ClusteringConfig(TypedDict):
    """Holds information about how to cluster trajectories."""

    do_histogram: bool
    n_bins: int
    do_knn: bool
    knn_ks: list[int]
    plot_all_trajectories: bool
    do_dbscan: bool
    dbscan_eps: int
    dbscan_minpt: int
    do_kmed: bool
    kmed_n_clusters: int


@final
class UserTrajectoryConfig(TypedDict, total=False):
    """Non clustering config."""

    coordinate_type_for_distance: CoordinateTypes
    n_rounds: int
    time: int
    side: SideSelection
    dtw: bool


@final
class UserClusteringConfig(TypedDict, total=False):
    """Non total TrajectoryConfig."""

    do_histogram: bool
    n_bins: int
    do_knn: bool
    knn_ks: list[int]
    plot_all_trajectories: bool
    do_dbscan: bool
    dbscan_eps: int
    dbscan_minpt: int
    do_kmed: bool
    kmed_n_clusters: int


@final
class DNNConfig(TypedDict):
    """Holds information about trajectories to analyze."""

    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    nodes_per_layer: int
    input_shape: tuple[int, ...]


@final
class UserDNNConfig(TypedDict, total=False):
    """Non clustering config."""

    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    nodes_per_layer: int


@final
class DNNTrajectoryConfig(TypedDict):
    """Holds information about trajectories to analyze."""

    coordinate_type: Literal["area", "token", "position"]
    side: SideSelection
    time: int
    consider_alive: bool


@final
class UserDNNTrajectoryConfig(TypedDict, total=False):
    """Non clustering config."""

    coordinate_type: str
    side: SideSelection
    time: int
    consider_alive: bool
