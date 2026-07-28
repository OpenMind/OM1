package qualityscorer

import (
	"regexp"
	"strings"

	"github.com/abadojack/whatlanggo"
	"golang.org/x/text/unicode/norm"

	"github.com/openmind/om1/internal/metrics"
)

// Best-effort language detection for short, often fragmentary trace text
// (ASR transcripts and TTS replies). Ported from dataAnalysis's
// scripts/lang_utils.py: a generic n-gram detector is unreliable on short,
// repetitive spoken fragments, so Chinese/Japanese/Korean are first checked
// deterministically by Unicode script (never left for the n-gram detector to
// confuse among), and pure-ASCII text containing a common English function
// word/contraction is classified as English directly. This is an
// approximation of the Python original, not a byte-identical port: no Go
// library shares langdetect's training corpus or quirks, so results on
// ambiguous short text will differ from historical data. This whole
// subsystem is explicitly best-effort, on both sides.

var (
	hangulRe = regexp.MustCompile(`[\x{AC00}-\x{D7A3}]`)
	kanaRe   = regexp.MustCompile(`[\x{3040}-\x{30FF}]`)
	hanRe    = regexp.MustCompile(`[\x{4E00}-\x{9FFF}\x{3400}-\x{4DBF}]`)

	englishWordRe = regexp.MustCompile(`[a-zA-Z']+`)
)

// englishStopwords mirrors lang_utils.py's ENGLISH_STOPWORDS set verbatim.
var englishStopwords = wordSet(`
a an the is isn't are aren't am was were wasn't weren't be been being
do does did don't doesn't didn't
i i'm i'll i've i'd you you're you'll you've you'd he he's she she's it it's
we we're we'll we've they they're they'll they've
me him her us them my your his its our their mine yours ours theirs
this that these those what what's who who's whom whose which
can can't could couldn't will won't would wouldn't shall should shouldn't
may might must
to of in on at for with without about into onto from by as
and or but if then than so because
yes no okay ok please thank thanks thank's hello hi hey bye goodbye
tell me more do you have has had have not just like maybe
here there here's there's where when why how
good bad nice great well know think want need going gonna
yeah yep yea sure let's shut up go away talk talking
`)

func wordSet(list string) map[string]struct{} {
	m := make(map[string]struct{})
	for _, w := range strings.Fields(list) {
		m[w] = struct{}{}
	}
	return m
}

// langNames maps common detected codes to human-readable names, mirroring
// lang_utils.py's LANG_NAMES. Anything not listed here is shown as its raw code.
var langNames = map[string]string{
	"en": "English", "es": "Spanish", "fr": "French", "de": "German",
	"it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
	"zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)", "zh": "Chinese",
	"ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
	"tr": "Turkish", "pl": "Polish", "vi": "Vietnamese", "th": "Thai",
	"sv": "Swedish", "id": "Indonesian", "el": "Greek", "he": "Hebrew",
	"da": "Danish", "fi": "Finnish", "no": "Norwegian", "cs": "Czech",
	"uk": "Ukrainian", "ro": "Romanian", "hu": "Hungarian",
	"ml": "Malayalam", "bn": "Bengali", "so": "Somali", "gu": "Gujarati",
	"et": "Estonian", "mk": "Macedonian", "sk": "Slovak", "te": "Telugu",
	"af": "Afrikaans", "ta": "Tamil", "ne": "Nepali", "sw": "Swahili",
	"tl": "Tagalog", "cy": "Welsh", "ca": "Catalan", "sq": "Albanian",
	"sl": "Slovenian", "lt": "Lithuanian",
}

// initLanguageLabels pre-registers every named language in langNames at zero
// on om1_quality_live_language_count, for the same reason StartServer calls
// metrics.InitQualityLabels() for classification/coherence: a label's very
// first real occurrence is otherwise invisible to increase()/rate() over
// short windows, since there's no prior sample for it to diff against (see
// metrics.InitQualityLabels's doc comment for the full explanation).
//
// This only covers the named subset, not every code detectLang could ever
// return -- an unmapped code (langName's raw-passthrough case) still hits
// that cold-start blind spot once, the first time it's ever seen, same as
// before this fix. That's an acceptable residual gap: this whole subsystem
// is already best-effort, and langNames covers every language anyone is
// realistically going to speak to the robot.
func initLanguageLabels() {
	for _, name := range langNames {
		metrics.QualityLiveLanguageCount.WithLabelValues(name).Add(0)
	}
}

func isASCII(s string) bool {
	for _, r := range s {
		if r >= 128 {
			return false
		}
	}
	return true
}

// looksEnglish mirrors lang_utils.py's looks_english: NFKC-normalize (folds
// full-width CJK punctuation to ASCII equivalents), reject anything non-ASCII,
// then check whether any lowercase word token is a common English function word.
func looksEnglish(text string) bool {
	normalized := norm.NFKC.String(text)
	if !isASCII(normalized) {
		return false
	}
	for _, w := range englishWordRe.FindAllString(strings.ToLower(normalized), -1) {
		if _, ok := englishStopwords[w]; ok {
			return true
		}
	}
	return false
}

// cjkScriptOverride mirrors lang_utils.py's cjk_script_override: Hangul,
// Kana, and Han-only text are resolved deterministically before any
// n-gram-based detector gets a chance to confuse among them. Returns "" if
// none apply.
func cjkScriptOverride(text string) string {
	if hangulRe.MatchString(text) {
		return "ko"
	}
	if kanaRe.MatchString(text) {
		return "ja"
	}
	if hanRe.MatchString(text) {
		return "zh-cn"
	}
	return ""
}

// detectLang returns a best-effort language code for text, or "" if
// undetectable -- mirroring lang_utils.py's detect_lang precedence exactly,
// substituting whatlanggo for langdetect as the final fallback (see the
// package doc comment above for why this isn't a byte-identical port).
func detectLang(text string) string {
	if lang := cjkScriptOverride(text); lang != "" {
		return lang
	}
	if looksEnglish(text) {
		return "en"
	}
	info := whatlanggo.Detect(text)
	if info.Lang < 0 {
		return ""
	}
	return info.Lang.Iso6391()
}

// langName maps a detected code to a human-readable name, mirroring
// lang_utils.py's lang_name -- unmapped codes pass through as-is.
func langName(code string) string {
	if name, ok := langNames[code]; ok {
		return name
	}
	return code
}
