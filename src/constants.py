import nltk
from nltk.corpus import stopwords

from pathlib import Path

nltk.download('stopwords')

ALL_STOPWORDS = set(stopwords.words("arabic") + stopwords.words("english"))

BASE_DIR = Path(__file__).parent.parent

DATA_PATH = BASE_DIR / "data"

TEST_DATA_PATH = DATA_PATH / "test.csv"

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

CLASSES_TRANSLATION = {
    "baby care": "العناية بالرضع",
    "bakery": "المخبوزات",
    "beef lamb meat": "لحم البقر والضأن",
    "beef processed meat": "لحم البقر واللحوم المصنعة",
    "biscuits cakes": "البسكويت والكيك",
    "candles air fresheners": "الشموع ومعطرات الجو",
    "chips crackers": "رقائق البطاطس والمقرمشات",
    "chips crackers": "رقائق البطاطس والمقرمشات",
    "chocolates sweets desserts": "الشوكولاتة والحلويات والحلويات الجاهزة",
    "cleaning supplies": "مستلزمات التنظيف",
    "condiments dressings marinades": "التوابل والصلصات والمخللات",
    "cooking ingredients": "مكونات الطبخ",
    "dairy eggs": "الألبان والبيض",
    "disposables napkins": "المنتجات أحادية الاستخدام والمناديل",
    "fruits": "الفواكه",
    "furniture": "الأثاث",
    "hair shower bath soap": "العناية بالشعر والاستحمام والصابون",
    "home appliances": "الأجهزة المنزلية",
    "home textile": "المنسوجات المنزلية",
    "jams spreads syrups": "المربى والدهن والشراب",
    "nuts dates dried fruits": "المكسرات والتمر والفواكه المجففة",
    "personal care skin body care": "العناية الشخصية والعناية بالبشرة والجسم",
    "poultry": "الدواجن",
    "rice pasta pulses": "الأرز والمعكرونة والبقوليات",
    "sauces dressings condiments": "الصلصات والتتبيلات والتوابل",
    "soft drinks juices": "المشروبات الغازية والعصائر",
    "sweets desserts": "الحلويات والتحليات",
    "tea and coffee": "الشاي والقهوة",
    "tea coffee hot drinks": "الشاي والقهوة والمشروبات الساخنة",
    "tins jars packets": "المعلبات والعبوات",
    "vegetables fruits": "الخضروات والفواكه",
    "vegetables herbs": "الخضروات والأعشاب",
    "water": "المياه",
    "laundry detergents": "الغسيل والمنظفات",
    "sugar home baking": "السكر والمخبوزات المنزلية",
    "perfumes deodorants": "العطور ومزيلات العرق",
    "stationary": "اللوازم المكتبية",
    "breakfast cereals bars": "ألواح حبوب الإفطار",
    "dental care": "العناية بالأسنان",
    "fish": "الأسماك",
    "footwear": "الأحذية",
    "mobile tablets": "الأجهزة اللوحية",
    "party supplies and gifts": "مستلزمات الحفلات والهدايا",
    "pets care": "العناية بالحيوانات الأليفة",
    "wear": "الملابس"
}