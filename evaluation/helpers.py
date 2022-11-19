def lang_code_to_name(language_code: str) -> str:
    """Converts language's ISO_639_1 code to language name
    Raises KeyError if language code is not one of the languages in the ted_talks_iwslt dataset"""
    ISO_639_1 = {
        "de": "German",
        "en": "English",
        "nl": "Dutch",
        "eu": "Basque",
        "ja": "Japanese",
        "ca": "Catalan",
        "fr-ca": "Canadian French",
        "hi": "Hindi",
    }
    return ISO_639_1[language_code]