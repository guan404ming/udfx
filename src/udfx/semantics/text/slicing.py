def mixed_slice_pattern(start: int, stop: int) -> str | None:
    if not -64 <= start < 0 or not 0 < stop <= 64:
        return None
    return rf"(?s)\A(?=.{{0,{stop}}}(.*)\z).*?(?=.{{0,{-start}}}\z)(.*)\1\z"
