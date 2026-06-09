package greeting_conversation

import "regexp"

// Time patterns: "11:00 a.m." -> "11 a.m.", "3:30 p.m." -> "3 30 p.m."
var (
	timeOnHour      = regexp.MustCompile(`\b(\d{1,2}):00\b`)
	timeWithMinutes = regexp.MustCompile(`\b(\d{1,2}):(\d{2})\b`)
)

// ttsCorrections expands abbreviations for cleaner TTS output.
var ttsCorrections = []struct {
	re   *regexp.Regexp
	repl string
}{
	// Month abbreviations
	{regexp.MustCompile(`\bJan\b`), "January"},
	{regexp.MustCompile(`\bFeb\b`), "February"},
	{regexp.MustCompile(`\bMar\b`), "March"},
	{regexp.MustCompile(`\bApr\b`), "April"},
	{regexp.MustCompile(`\bJun\b`), "June"},
	{regexp.MustCompile(`\bJul\b`), "July"},
	{regexp.MustCompile(`\bAug\b`), "August"},
	{regexp.MustCompile(`\bSep(?:t)?\b`), "September"},
	{regexp.MustCompile(`\bOct\b`), "October"},
	{regexp.MustCompile(`\bNov\b`), "November"},
	{regexp.MustCompile(`\bDec\b`), "December"},
	// Address abbreviations
	{regexp.MustCompile(`\bSt\b\.?`), "Street"},
	{regexp.MustCompile(`\bAve\b\.?`), "Avenue"},
	{regexp.MustCompile(`\bBlvd\b\.?`), "Boulevard"},
	{regexp.MustCompile(`\bDr\b\.?`), "Drive"},
	{regexp.MustCompile(`\bRd\b\.?`), "Road"},
	{regexp.MustCompile(`\bLn\b\.?`), "Lane"},
	{regexp.MustCompile(`\bCt\b\.?`), "Court"},
	{regexp.MustCompile(`\bPl\b\.?`), "Place"},
	{regexp.MustCompile(`\bPkwy\b\.?`), "Parkway"},
	{regexp.MustCompile(`\bHwy\b\.?`), "Highway"},
	// Directional abbreviations (capture the following token in place of a lookahead)
	{regexp.MustCompile(`\bN\b\.?(\s+[A-Z])`), "North${1}"},
	{regexp.MustCompile(`\bS\b\.?(\s+[A-Z])`), "South${1}"},
	{regexp.MustCompile(`\bE\b\.?(\s+[A-Z])`), "East${1}"},
	{regexp.MustCompile(`\bW\b\.?(\s+[A-Z])`), "West${1}"},
}

// normalizeTTSText applies regex corrections to expand common abbreviations in TTS text.
func normalizeTTSText(text string) string {
	text = timeOnHour.ReplaceAllString(text, "${1}")
	text = timeWithMinutes.ReplaceAllString(text, "${1} ${2}")

	for _, c := range ttsCorrections {
		text = c.re.ReplaceAllString(text, c.repl)
	}

	return text
}
