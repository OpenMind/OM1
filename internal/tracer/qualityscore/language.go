package qualityscore

import (
	"regexp"
	"strings"

	"github.com/abadojack/whatlanggo"
	"golang.org/x/text/unicode/norm"

	"github.com/openmind/om1/internal/metrics"
)

var (
	hangulRe = regexp.MustCompile(`[\x{AC00}-\x{D7A3}]`)
	kanaRe   = regexp.MustCompile(`[\x{3040}-\x{30FF}]`)
	hanRe    = regexp.MustCompile(`[\x{4E00}-\x{9FFF}\x{3400}-\x{4DBF}]`)

	englishWordRe = regexp.MustCompile(`[a-zA-Z']+`)
)

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

func langName(code string) string {
	if name, ok := langNames[code]; ok {
		return name
	}
	return code
}
