"""Explicit ordering shared by training, calibration, and reporting."""

# name, VoxLingua ISO code, FLEURS configuration
LANGUAGES = (
    ("Arabic", "ar", "ar_eg"),
    ("English", "en", "en_us"),
    ("Spanish", "es", "es_419"),
    ("Persian", "fa", "fa_ir"),
    ("Hindi", "hi", "hi_in"),
    ("Indonesian", "id", "id_id"),
    ("Maori", "mi", "mi_nz"),
    ("Pushto", "ps", "ps_af"),
    ("Russian", "ru", "ru_ru"),
    ("Thai", "th", "th_th"),
    ("Urdu", "ur", "ur_pk"),
    ("Mandarin Chinese", "zh", "cmn_hans_cn"),
)
CODES = [item[1] for item in LANGUAGES]
NAMES = [item[0] for item in LANGUAGES]


def selected_indices(labels):
    by_code = {label.split(":", 1)[0].strip(): i for i, label in enumerate(labels)}
    missing = set(CODES) - by_code.keys()
    if missing:
        raise ValueError(
            f"Pretrained model is missing language codes: {sorted(missing)}"
        )
    return [by_code[code] for code in CODES]
