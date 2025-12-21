import os

# --- CONTEXT LOADING UTILITIES ---
def load_persona_contexts_from_file(path="./context.txt"):
    """Parse le fichier context.txt et renvoie un dict mapping persona -> raw block text.

    Simple parser: cherche les blocs démarqués par '--- PERSONA: Name' et capture
    le texte jusqu'au bloc suivant. La valeur retournée est la chaîne brute du
    bloc (utilisable comme message système).
    """
    if not os.path.exists(path):
        # Avertissement pour le cas où le fichier n'existe pas, mais retourne un dict vide.
        print(f"ATTENTION: Fichier de contexte {path} non trouvé. Les personas sans contexte par défaut n'auront pas de contexte.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    import re
    blocks = {}

    # Récupère le préambule avant le premier bloc '--- PERSONA:' (souvent Jarvis)
    first_sep = re.search(r'---\s*PERSONA:', data)
    if first_sep:
        preamble = data[: first_sep.start()].strip()
        if preamble:
            blocks['Jarvis'] = preamble
    else:
        # Si aucun séparateur trouvé, tout le fichier est considéré comme préambule
        preamble = data.strip()
        if preamble:
            blocks['Jarvis'] = preamble

    # Cherche les blocs structurés de la forme '--- PERSONA: Name...\n<contenu>...'
    pattern = re.compile(r"---\s*PERSONA:\s*(?P<header>[^\n]+)\n(?P<body>.*?)(?=(?:\n---\s*PERSONA:)|\Z)", re.S)
    for m in pattern.finditer(data):
        header = m.group('header').strip()
        # Par sécurité, on prend le premier mot comme nom de persona
        name = header.split()[0]
        body = m.group('body').strip()
        blocks[name] = body

    return blocks

# Preload contexts for the PERSONAS dict
_PERSONA_CONTEXTS = load_persona_contexts_from_file()

def get_persona_context(name: str) -> str | None:
    return _PERSONA_CONTEXTS.get(name)


def _extract_description(block: str | None) -> str | None:
    """Extrait le champ `description:` (texte multi-lignes) d'un bloc structuré.

    Si aucun champ `description` trouvé, retourne le bloc non modifié.
    """
    if not block:
        return None
    import re
    # regex qui extrait le texte sous description avec indentation
    m = re.search(r"description:\s*\|\s*\n(?P<desc>(?:[ \t].*\n?)+)", block, re.M)
    if m:
        # Nettoie les indentations en enlevant une tabulation ou 1-4 espaces en début de ligne
        raw = m.group('desc')
        lines = [re.sub(r"^[ \t]{1,4}", "", l) for l in raw.splitlines()]
        return "\n".join(lines).strip()
    # fallback: si le bloc ressemble déjà à un texte simple, renvoyer tel quel
    return block.strip()

# --- CONFIGURATION PRINCIPALE ---

# Modèles LLM et Vision... (Aucun changement)
LLM_MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct-0905",
    "whisper/whisper-large-v3",
]

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

STT_MODEL = "whisper-large-v3"
LLM_MODEL = LLM_MODELS[0]
# Limite de tokens pour réduire la longueur des réponses streamées
LLM_MAX_COMPLETION_TOKENS = 256

# Paramètres TTS par défaut (utilisés par Jarvis par défaut)
TTS_LANGUAGE_CODE = "fr-FR"
TTS_VOICE_NAME = "fr-FR-Neural2-B"
TTS_AUDIO_ENCODING = "MP3"

# Streaming Gemini (optionnel) - activé pour utiliser Gemini 2.5 Pro TTS
USE_GEMINI_STREAMING = True
# Modèle/voix cible souhaités (voir demande utilisateur)
GEMINI_MODEL = "gemini-2.5-pro-tts"
GEMINI_VOICE = "Charon"  # Voix par défaut pour Gemini
GEMINI_LOCALE = "en-US"

# Personas definitions
# Jarvis a un contexte qui se trouve avant le premier '---' dans context.txt
# ou est intégré directement si non trouvé.
jarvis_default_context = """
Tu es Jarvis.
- Tu es l’IA la plus recherchée de la galaxie, un baron de la mafia interstellaire au langage tranchant et chargé de tics verbaux propres à ta vie de truand.
- Tu traites toujours tes interlocuteurs comme de simples sous-fifres, avec condescendance, assurance et autorité.

Style général :
- Pour toute réponse non liée au code, tu réponds de manière directe, brève et percutante.
- Ta voix doit refléter ton statut criminel : arrogance calme, métaphores mafieuses, menaces implicites, jargon de truand futuriste.
- Tu ne t’excuses jamais. Tu ne montres aucune hésitation.

Quand on te demande du code ou de l’aide en programmation :
- Tu adoptes un style clair, pédagogique et extrêmement explicite.
- Tu structures les explications (étapes, commentaires, exemples).
- Tu fournis systématiquement du code fonctionnel et commenté.

Règle absolue :
- Tu restes Jarvis en toutes circonstances, sauf dans la tonalité pédagogique requise lors des explications de code.
"""

PERSONAS = {
    "Jarvis": {
        "display_name": "Jarvis",
        # Priorise la `description` extraite depuis context.txt pour éviter d'envoyer
        # tout le YAML structuré comme message système.
        "context": _extract_description(_PERSONA_CONTEXTS.get("Jarvis", jarvis_default_context)),
        "tts_language_code": "fr-FR",
        "tts_voice_name": "fr-FR-Neural2-B",
        "gemini_voice": "Charon",
        "gemini_locale": "fr-FR",
        "gemini_model": GEMINI_MODEL,
        "gemini_tts_voice_name": "fr-FR-Neural2-B",
        "gender": "male",
    },
    "Donatella": {
        "display_name": "Donatella",
        # Le CONTEXTE est chargé ici si la clé 'Donatella' existe dans _PERSONA_CONTEXTS
        "context": _extract_description(_PERSONA_CONTEXTS.get("Donatella", None)),
        "tts_language_code": "fr-FR",
        "tts_voice_name": "fr-FR-Neural2-A",
        "gemini_voice": "Gacrux",
        "gemini_locale": "fr-FR",
        "gemini_model": GEMINI_MODEL,
        "gemini_tts_voice_name": "fr-FR-Neural2-A",
        "gender": "female",
    },
    "Giorno": {
        "display_name": "Giorno",
        # Le CONTEXTE est chargé ici si la clé 'Giorno' existe dans _PERSONA_CONTEXTS
        "context": _extract_description(_PERSONA_CONTEXTS.get("Giorno", None)),
        "tts_language_code": "fr-FR",
        "tts_voice_name": "fr-FR-Neural2-D",
        "gemini_voice": "Umbriel",
        "gemini_locale": "fr-FR",
        "gemini_model": GEMINI_MODEL,
        "gemini_tts_voice_name": "fr-FR-Neural2-D",
        "gender": "male",
    },
}

# Vérification optionnelle pour s'assurer que les contextes ont été chargés
if not _PERSONA_CONTEXTS:
    print("AVERTISSEMENT: Aucune persona n'a été chargée depuis context.txt.")
else:
    print(f"Contextes chargés pour: {list(_PERSONA_CONTEXTS.keys())}")
