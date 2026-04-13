"""
breed_labels.py
---------------
Combined breed label mapping from:
  - Oxford-IIIT Pet Dataset (37 breeds: 25 dogs + 12 cats)
  - Stanford Dogs Dataset (120 dog breeds)
  - ImageNet synset IDs for mapping EfficientNet-B0 predictions

These are the breeds the system knows about. The embedding model
(EfficientNet-B0, pretrained on ImageNet) recognises most of these
directly via its top-1000 class output.

We use two strategies:
  1. Direct classification: map ImageNet class → breed label (for supported breeds)
  2. Embedding comparison: cosine similarity between 1280-dim feature vectors
     for breeds not in ImageNet top-1000 (or when confidence is low)
"""

# ─── Oxford-IIIT Pet Dataset Breeds ──────────────────────────────────────────
OXFORD_DOG_BREEDS = [
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Boxer",
    "Chihuahua",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired Pointer",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Miniature Pinscher",
    "Newfoundland",
    "Pomeranian",
    "Pug",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
]

OXFORD_CAT_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
]

# ─── Stanford Dogs Dataset Breeds (120 breeds) ───────────────────────────────
STANFORD_DOG_BREEDS = [
    "Affenpinscher", "Afghan Hound", "African Hunting Dog", "Airedale",
    "American Staffordshire Terrier", "Appenzeller", "Australian Terrier",
    "Basenji", "Basset", "Beagle", "Bedlington Terrier", "Bernese Mountain Dog",
    "Black and Tan Coonhound", "Blenheim Spaniel", "Bloodhound", "Bluetick",
    "Border Collie", "Border Terrier", "Borzoi", "Boston Bull",
    "Bouvier des Flandres", "Boxer", "Brabancon Griffon", "Briard",
    "Brittany Spaniel", "Bull Mastiff", "Cairn", "Cardigan",
    "Chesapeake Bay Retriever", "Chihuahua", "Chow", "Clumber",
    "Cocker Spaniel", "Collie", "Curly-Coated Retriever", "Dandie Dinmont",
    "Dhole", "Dingo", "Doberman", "English Foxhound", "English Setter",
    "English Springer", "EntleBucher", "Eskimo Dog", "Flat-Coated Retriever",
    "French Bulldog", "German Shepherd", "German Short-Haired Pointer",
    "Giant Schnauzer", "Golden Retriever", "Gordon Setter", "Great Dane",
    "Great Pyrenees", "Greater Swiss Mountain Dog", "Groenendael",
    "Ibizan Hound", "Irish Setter", "Irish Terrier", "Irish Water Spaniel",
    "Irish Wolfhound", "Italian Greyhound", "Japanese Spaniel",
    "Kerry Blue Terrier", "Kelpie", "Komondor", "Kuvasz",
    "Labrador Retriever", "Lakeland Terrier", "Leonberg", "Lhasa",
    "Malamute", "Malinois", "Maltese Dog", "Mexican Hairless",
    "Miniature Pinscher", "Miniature Poodle", "Miniature Schnauzer",
    "Norfolk Terrier", "Norwegian Elkhound", "Norwich Terrier",
    "Old English Sheepdog", "Otterhound", "Papillon", "Pekinese",
    "Pembroke", "Pomeranian", "Pug", "Redbone", "Rhodesian Ridgeback",
    "Rottweiler", "Saint Bernard", "Saluki", "Samoyed", "Schipperke",
    "Scotch Terrier", "Scottish Deerhound", "Sealyham Terrier", "Shetland Sheepdog",
    "Shih-Tzu", "Siberian Husky", "Silky Terrier", "Soft-Coated Wheaten Terrier",
    "Staffordshire Bullterrier", "Standard Poodle", "Standard Schnauzer",
    "Sussex Spaniel", "Tibetan Mastiff", "Tibetan Terrier",
    "Toy Manchester Terrier", "Toy Poodle", "Toy Terrier",
    "Vizsla", "Walker Hound", "Weimaraner", "Welsh Springer Spaniel",
    "West Highland White Terrier", "Whippet", "Wire-Haired Fox Terrier",
    "Yorkshire Terrier",
]

# ─── ImageNet class index → breed name mapping ───────────────────────────────
# EfficientNet-B0 outputs 1000 classes. Dog breeds occupy indices 151-268.
# Cats occupy indices 281-285. This maps those to our breed labels.
# Source: ImageNet synset labels (ILSVRC 2012)
IMAGENET_DOG_INDICES = {
    151: "Chihuahua", 152: "Japanese Chin", 153: "Maltese Dog",
    154: "Pekinese", 155: "Shih-Tzu", 156: "Toy Terrier",
    157: "Rhodesian Ridgeback", 158: "Afghan Hound", 159: "Basset",
    160: "Beagle", 161: "Bloodhound", 162: "Bluetick", 163: "Black and Tan Coonhound",
    164: "Walker Hound", 165: "English Foxhound", 166: "Redbone",
    167: "Borzoi", 168: "Irish Wolfhound", 169: "Italian Greyhound",
    170: "Whippet", 171: "Ibizan Hound", 172: "Norwegian Elkhound",
    173: "Otterhound", 174: "Saluki", 175: "Scottish Deerhound",
    176: "Weimaraner", 177: "Staffordshire Bullterrier",
    178: "American Staffordshire Terrier", 179: "Bedlington Terrier",
    180: "Border Terrier", 181: "Kerry Blue Terrier", 182: "Irish Terrier",
    183: "Norfolk Terrier", 184: "Yorkshire Terrier", 185: "Wire-Haired Fox Terrier",
    186: "Lakeland Terrier", 187: "Sealyham Terrier", 188: "Airedale",
    189: "Cairn", 190: "Australian Terrier", 191: "Dandie Dinmont",
    192: "Boston Bull", 193: "Miniature Schnauzer", 194: "Giant Schnauzer",
    195: "Standard Schnauzer", 196: "Scotch Terrier", 197: "Tibetan Terrier",
    198: "Silky Terrier", 199: "Soft-Coated Wheaten Terrier",
    200: "West Highland White Terrier", 201: "Lhasa",
    202: "Flat-Coated Retriever", 203: "Curly-Coated Retriever",
    204: "Golden Retriever", 205: "Labrador Retriever",
    206: "Chesapeake Bay Retriever", 207: "German Short-Haired Pointer",
    208: "Vizsla", 209: "English Setter", 210: "Irish Setter",
    211: "Gordon Setter", 212: "Brittany Spaniel", 213: "Clumber",
    214: "English Springer", 215: "Welsh Springer Spaniel",
    216: "Cocker Spaniel", 217: "Sussex Spaniel", 218: "Irish Water Spaniel",
    219: "Kuvasz", 220: "Schipperke", 221: "Groenendael", 222: "Malinois",
    223: "Briard", 224: "Kelpie", 225: "Komondor", 226: "Old English Sheepdog",
    227: "Shetland Sheepdog", 228: "Collie", 229: "Border Collie",
    230: "Bouvier des Flandres", 231: "Rottweiler", 232: "German Shepherd",
    233: "Doberman", 234: "Miniature Pinscher", 235: "Greater Swiss Mountain Dog",
    236: "Bernese Mountain Dog", 237: "AppenzEller", 238: "EntleBucher",
    239: "Boxer", 240: "Bull Mastiff", 241: "Tibetan Mastiff",
    242: "French Bulldog", 243: "Great Dane", 244: "Saint Bernard",
    245: "Eskimo Dog", 246: "Malamute", 247: "Siberian Husky",
    248: "Dalmatian", 249: "Affenpinscher", 250: "Brabancon Griffon",
    251: "Papillon", 252: "Toy Poodle", 253: "Miniature Poodle",
    254: "Standard Poodle", 255: "Mexican Hairless", 256: "Dingo",
    257: "Dhole", 258: "African Hunting Dog", 259: "Hyena",
    260: "Red Wolf", 261: "Coyote", 262: "Timber Wolf",
    263: "White Wolf", 264: "Red Fox", 265: "Kit Fox",
    266: "Arctic Fox", 267: "Grey Fox", 268: "Pomeranian",
}

IMAGENET_CAT_INDICES = {
    281: "Tabby Cat", 282: "Tiger Cat", 283: "Persian",
    284: "Siamese", 285: "Egyptian Mau",
}

# Combined lookup
IMAGENET_PET_INDICES = {**IMAGENET_DOG_INDICES, **IMAGENET_CAT_INDICES}

# ─── Species lookup ───────────────────────────────────────────────────────────
CAT_BREEDS_SET = set(OXFORD_CAT_BREEDS + ["Tabby Cat", "Tiger Cat", "Egyptian Mau"])
DOG_INDICES_SET = set(IMAGENET_DOG_INDICES.keys())
CAT_INDICES_SET = set(IMAGENET_CAT_INDICES.keys())

def get_species(imagenet_idx: int) -> str:
    if imagenet_idx in CAT_INDICES_SET:
        return "cat"
    if imagenet_idx in DOG_INDICES_SET:
        return "dog"
    return "unknown"

def get_all_breeds() -> list:
    """Return deduplicated list of all known breeds."""
    all_breeds = set(OXFORD_DOG_BREEDS + OXFORD_CAT_BREEDS + STANFORD_DOG_BREEDS)
    return sorted(all_breeds)
