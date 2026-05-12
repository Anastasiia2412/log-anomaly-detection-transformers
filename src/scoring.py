def extract_key_events(row, max_items=8):
    items = []

    event_ids = row.get("EventIDs", [])
    sources = row.get("Sources", [])
    levels = row.get("Levels", [])

    for event_id in event_ids:
        items.append(f"EventID {event_id}")

    for source in sources:
        items.append(str(source))

    for level in levels:
        if str(level).lower() in ["error", "critical", "warning"]:
            items.append(str(level))

    unique = []

    for item in items:
        if item not in unique:
            unique.append(item)

    return unique[:max_items]
