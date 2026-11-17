"""
The 151 original Kanto Pokemon, in Pokedex order (Bulbasaur #1 -> Mew #151).

Why this file exists: our dataset will come from a Kaggle download that
might include Pokemon from other generations, weird capitalization, or
alternate spellings. Having a canonical, known-good list lets prepare_data.py
filter down to exactly the Pokemon we want, and lets every other script
agree on the same set of class names.

When you're ready to "upscale" (per your original request), you extend this
list -- everything downstream (prepare_data.py, dataset.py, model.py,
train.py) reads the number of classes dynamically, so nothing else needs to
change.
"""

KANTO_151 = [
    "Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard",
    "Squirtle", "Wartortle", "Blastoise", "Caterpie", "Metapod", "Butterfree",
    "Weedle", "Kakuna", "Beedrill", "Pidgey", "Pidgeotto", "Pidgeot", "Rattata",
    "Raticate", "Spearow", "Fearow", "Ekans", "Arbok", "Pikachu", "Raichu",
    "Sandshrew", "Sandslash", "Nidoran-f", "Nidorina", "Nidoqueen", "Nidoran-m",
    "Nidorino", "Nidoking", "Clefairy", "Clefable", "Vulpix", "Ninetales",
    "Jigglypuff", "Wigglytuff", "Zubat", "Golbat", "Oddish", "Gloom", "Vileplume",
    "Paras", "Parasect", "Venonat", "Venomoth", "Diglett", "Dugtrio", "Meowth",
    "Persian", "Psyduck", "Golduck", "Mankey", "Primeape", "Growlithe", "Arcanine",
    "Poliwag", "Poliwhirl", "Poliwrath", "Abra", "Kadabra", "Alakazam", "Machop",
    "Machoke", "Machamp", "Bellsprout", "Weepinbell", "Victreebel", "Tentacool",
    "Tentacruel", "Geodude", "Graveler", "Golem", "Ponyta", "Rapidash", "Slowpoke",
    "Slowbro", "Magnemite", "Magneton", "Farfetchd", "Doduo", "Dodrio", "Seel",
    "Dewgong", "Grimer", "Muk", "Shellder", "Cloyster", "Gastly", "Haunter",
    "Gengar", "Onix", "Drowzee", "Hypno", "Krabby", "Kingler", "Voltorb",
    "Electrode", "Exeggcute", "Exeggutor", "Cubone", "Marowak", "Hitmonlee",
    "Hitmonchan", "Lickitung", "Koffing", "Weezing", "Rhyhorn", "Rhydon",
    "Chansey", "Tangela", "Kangaskhan", "Horsea", "Seadra", "Goldeen", "Seaking",
    "Staryu", "Starmie", "Mr-mime", "Scyther", "Jynx", "Electabuzz", "Magmar",
    "Pinsir", "Tauros", "Magikarp", "Gyarados", "Lapras", "Ditto", "Eevee",
    "Vaporeon", "Jolteon", "Flareon", "Porygon", "Omanyte", "Omastar", "Kabuto",
    "Kabutops", "Aerodactyl", "Snorlax", "Articuno", "Zapdos", "Moltres",
    "Dratini", "Dragonair", "Dragonite", "Mewtwo", "Mew",
]

assert len(KANTO_151) == 151, f"Expected 151 Pokemon, got {len(KANTO_151)}"

# Lowercased, alnum-only lookup for fuzzy-matching folder/file names coming
# from whatever dataset you download (e.g. "Mr. Mime", "mr-mime", "MrMime"
# all normalize to "mrmime").
def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())

NORMALIZED_TO_CANONICAL = {_normalize(name): name for name in KANTO_151}


def match_kanto_name(raw_name: str):
    """Return the canonical Kanto name for a raw folder/file name, or None."""
    return NORMALIZED_TO_CANONICAL.get(_normalize(raw_name))
