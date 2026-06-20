from __future__ import annotations

_CROWD_CAUSES       = {"public_event", "procession", "protest", "vip_movement"}
_INFRA_CAUSES       = {"construction", "water_logging", "tree_fall"}

_CROWD_MANPOWER     = 4;  _CROWD_BARRICADES    = 8
_INFRA_MANPOWER     = 2;  _INFRA_BARRICADES    = 6
_HIGH_PRI_MANPOWER  = 4;  _HIGH_PRI_BARRICADES = 5
_CLOSURE_MANPOWER   = 6;  _CLOSURE_BARRICADES  = 15
_BASE_MANPOWER      = 2;  _BASE_BARRICADES     = 0


def calculate_optimal_manpower_and_hardware(
    priority_pred: str,
    event_cause: str,
    requires_closure: bool
) -> dict:
    """
    Analytically compute required manpower and barricades.

    Parameters
    ----------
    priority_pred    : 'High' or 'Low'
    event_cause      : raw event cause string from dataset
    requires_closure : boolean flag

    Returns
    -------
    dict with keys: manpower, barricades, breakdown
    """
    manpower  = _BASE_MANPOWER
    barricades = _BASE_BARRICADES
    breakdown  = {"base": {"manpower": manpower, "barricades": barricades}}

    cause_norm = str(event_cause).strip().lower().replace(" ", "_")

    if str(priority_pred).strip().lower() == "high":
        manpower   += _HIGH_PRI_MANPOWER
        barricades += _HIGH_PRI_BARRICADES
        breakdown["high_priority"] = {"manpower": _HIGH_PRI_MANPOWER,
                                       "barricades": _HIGH_PRI_BARRICADES}

    if requires_closure:
        manpower   += _CLOSURE_MANPOWER
        barricades += _CLOSURE_BARRICADES
        breakdown["road_closure"] = {"manpower": _CLOSURE_MANPOWER,
                                      "barricades": _CLOSURE_BARRICADES}

    if cause_norm in _CROWD_CAUSES or event_cause in _CROWD_CAUSES:
        manpower   += _CROWD_MANPOWER
        barricades += _CROWD_BARRICADES
        breakdown["crowd_event"] = {"manpower": _CROWD_MANPOWER,
                                     "barricades": _CROWD_BARRICADES}
    elif cause_norm in _INFRA_CAUSES or event_cause in _INFRA_CAUSES:
        manpower   += _INFRA_MANPOWER
        barricades += _INFRA_BARRICADES
        breakdown["infrastructure_event"] = {"manpower": _INFRA_MANPOWER,
                                              "barricades": _INFRA_BARRICADES}

    return {
        "manpower":   manpower,
        "barricades": barricades,
        "breakdown":  breakdown,
    }


def format_resource_summary(resource_dict: dict) -> str:
    lines = [
        f"  Total Manpower  : {resource_dict['manpower']} personnel",
        f"  Total Barricades: {resource_dict['barricades']} units",
        "",
        "  Breakdown:",
    ]
    for label, vals in resource_dict["breakdown"].items():
        lines.append(f"    [{label}]  +{vals['manpower']} manpower  "
                     f"+{vals['barricades']} barricades")
    return "\n".join(lines)


if __name__ == "__main__":
    result = calculate_optimal_manpower_and_hardware("High", "procession", True)
    print(result)
    print(format_resource_summary(result))
