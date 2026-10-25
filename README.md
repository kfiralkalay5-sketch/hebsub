# hebsub – תרגום כתוביות מאנגלית לעברית, מקומית על המחשב

hebsub מתרגם כתוביות באנגלית של סרטים וסדרות לכתוביות בעברית. התרגום רץ על המחשב שלכם, עם מודל שפה מקומי דרך [Ollama](https://ollama.com), בלי לשלם על שירות ענן.

## מה יש בתרגום
- **זכר ונקבה:**
  - רשימת דמויות לכל פרק, ופנייה בזכר או בנקבה לפי מי מדבר ואל מי.
  - מעבר נוסף שבודק את הצורות האלה.
- **ניקוי של כתוביות המקור:**
  - תיאורי צלילים ושמות דוברים, כמו `[sighs]` ו-`[Name]`.
  - קרדיטים של קבוצות תרגום.
  - תיאורי מוזיקה.
- **עקביות:**
  - שמות ומונחים כתובים אותו דבר לאורך כל הסדרה.
  - זיכרון דמויות לכל סדרה.
- **בדיקה ועריכה:**
  - בדיקה של כל המשפטים מול האנגלית.
  - תיקון של שורות קטועות ושל מילים באנגלית שנשארו.
- **פורמט:** פלט SRT עם כיווניות מימין לשמאל, ובדיוק אותם זמנים כמו באנגלית.

איכות: התרגום מובן וזורם, אבל עדיין לא ברמה של מתרגם מקצועי. בבדיקות עיוורות מצאו בו בערך פי 2 עד 3 יותר טעויות מאשר בתרגום אנושי.

## דרישות
| רכיב | מינימום |
|---|---|
| מערכת הפעלה | Windows 10/11, macOS (Apple Silicon), Linux |
| Python | 3.10 ומעלה |
| זיכרון למודל המומלץ (Gemma 4 26B) | כרטיס מסך עם 20GB לפחות (למשל RTX 3090/4090), או מק עם 32GB זיכרון |
| מחשב חלש יותר | אפשר מודל קטן יותר (ראו למטה), באיכות נמוכה יותר |
| ffmpeg | נדרש כדי לחלץ כתוביות באנגלית מתוך קובץ הווידאו |

## התקנה

### Windows (PowerShell)
```powershell
winget install Ollama.Ollama
winget install Python.Python.3.12
winget install Gyan.FFmpeg
# פתחו PowerShell חדש אחרי ההתקנות
ollama pull hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
git clone <כתובת המאגר> hebsub
cd hebsub
python -m pip install -e .
hebsub check
```

### macOS
```bash
brew install ollama ffmpeg python@3.12
ollama pull hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
git clone <כתובת המאגר> hebsub && cd hebsub
python3 -m pip install -e .
hebsub check
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo apt install ffmpeg python3-pip   # או המקבילה בהפצה שלכם
ollama pull hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
git clone <כתובת המאגר> hebsub && cd hebsub
python3 -m pip install -e .
hebsub check
```

## שימוש
```bash
hebsub translate "D:\TV\Show\Season 1\Show.S01E01.mkv"     # פרק אחד
hebsub scan "D:\TV\Show" --translate                       # כל מה שאין לו עברית בתיקייה
hebsub check                                               # בדיקת התקנה וחיבור למודל
```
- **מאיפה מגיעות הכתוביות באנגלית:** המתרגם מחפש קובץ באנגלית ליד הווידאו, למשל `Show.S01E01.en.srt`. אם אין כזה, הוא מחלץ את הכתוביות מתוך הווידאו עם ffmpeg.
- **מה נוצר:** קובץ `Show.S01E01.he.srt` ליד הווידאו.
- **כתוביות קיימות:** הן לא נדרסות. אם כבר יש כתוביות בעברית, הפרק מדולג (אלא אם מוסיפים `--force`).

## הגדרות
- **בלי הגדרות בכלל:** הכלי עובד עם ברירות המחדל שב-`hebsub/default_config.json`.
- **כדי לשנות:** מעתיקים את `config.example.json` ל-`config.json` בתיקיית הנתונים ועורכים אותו. אפשר גם להעביר נתיב עם `--config`.
- **איפה תיקיית הנתונים:**
  - Windows: `%APPDATA%\hebsub`
  - macOS: `~/Library/Application Support/hebsub`
  - Linux: `~/.local/share/hebsub`
  - אפשר לשנות אותה עם משתנה הסביבה `HEBSUB_HOME`.
- **מודל קטן יותר:** משנים את `ollama.model`, למשל ל-`gemma3:12b`. צריך כ-10GB, והאיכות נמוכה יותר.
- **מפתחות API:** לא נשמרים בקובץ ההגדרות. שמים אותם במשתני סביבה. ב-Windows: `setx TMDB_API_KEY "..."`, ובמק ובלינוקס: `export ...`.

## בקרוב
- **גרסה 0.2:**
  - פקודה `hebsub setup` שמחברת את Sonarr, Radarr ו-Bazarr.
  - מצב `hebsub watch` שמתרגם הורדות חדשות אוטומטית.
  - מפתח TMDB לזיהוי מגדר הדמויות.
  - מודל קול אופציונלי (זיהוי דובר וגבר או אישה), שרץ על המעבד בכל מערכת הפעלה.
- **גרסה 0.3:** למידה אוטומטית:
  - אוצר מילים לכל סדרה, מכתוביות עבריות אנושיות שיש בספרייה.
  - זיכרון תרגום.
  - לימוד מתיקונים שהמשתמש עושה בקבצים.

## רישיונות
- **CueIvrit:** הכלים בתיקייה `hebsub/vendor/cueivrit` (הכנה, בנייה ובדיקה של קבצי SRT) הם CueIvrit מאת Guy Tsarfati, ברישיון MIT. הרישיון המלא נמצא בתיקייה.
- **המודלים:** יש להם רישיונות משלהם. Gemma מופץ בתנאי השימוש של Google.
