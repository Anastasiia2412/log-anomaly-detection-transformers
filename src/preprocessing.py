import re


def normalize_message(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<guid>",
        text,
    )

    text = re.sub(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "<ip>",
        text,
    )

    text = re.sub(
        r"[a-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*",
        "<path>",
        text,
    )

    text = re.sub(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "<date>",
        text,
    )

    text = re.sub(
        r"\b\d{1,2}:\d{2}(:\d{2})?\b",
        "<time>",
        text,
    )

    text = re.sub(
        r"\b0x[0-9a-f]+\b",
        "<hex>",
        text,
    )

    text = re.sub(
        r"\b\d+\b",
        "<num>",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_event_tokens(df):
    df = df.copy()

    df["NormalizedMessage"] = df["Message"].apply(normalize_message)

    df["EventToken"] = df.apply(
        lambda row: (
            f"{row['LogName']} | "
            f"{row['Source']} | "
            f"{row['Event ID']} | "
            f"{row['Level']}"
        ),
        axis=1,
    )

    df["EventTemplateForReview"] = df.apply(
        lambda row: (
            f"{row['LogName']} | "
            f"{row['Source']} | "
            f"{row['Event ID']} | "
            f"{row['Level']} | "
            f"{row['NormalizedMessage']}"
        ),
        axis=1,
    )

    return df
