package vlm

import (
	"bufio"
	"context"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

// VideoDevice describes an available video capture device.
type VideoDevice struct {
	Index int
	Name  string
}

var avfoundationVideoLine = regexp.MustCompile(`\[(\d+)\]\s+(.+)$`)

func EnumerateVideoDevices() []VideoDevice {
	switch runtime.GOOS {
	case "linux":
		return enumerateLinuxVideoDevices()
	case "darwin":
		return enumerateDarwinVideoDevices()
	default:
		return nil
	}
}

func enumerateLinuxVideoDevices() []VideoDevice {
	matches, err := filepath.Glob("/dev/video*")
	if err != nil {
		return nil
	}

	var devices []VideoDevice
	for _, path := range matches {
		suffix := strings.TrimPrefix(filepath.Base(path), "video")
		idx, err := strconv.Atoi(suffix)
		if err != nil {
			continue
		}
		devices = append(devices, VideoDevice{Index: idx, Name: path})
	}
	sort.Slice(devices, func(i, j int) bool { return devices[i].Index < devices[j].Index })

	return devices
}

func enumerateDarwinVideoDevices() []VideoDevice {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-f", "avfoundation",
		"-list_devices", "true",
		"-i", "",
	)
	out, _ := cmd.CombinedOutput()
	return parseAVFoundationDevices(string(out))
}

func parseAVFoundationDevices(output string) []VideoDevice {
	var devices []VideoDevice
	scanner := bufio.NewScanner(strings.NewReader(output))
	inVideoSection := false

	for scanner.Scan() {
		line := scanner.Text()
		lower := strings.ToLower(line)
		if strings.Contains(lower, "avfoundation video devices") {
			inVideoSection = true
			continue
		}

		if strings.Contains(lower, "avfoundation audio devices") {
			inVideoSection = false
			continue
		}

		if !inVideoSection {
			continue
		}

		m := avfoundationVideoLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}

		idx, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}

		devices = append(devices, VideoDevice{Index: idx, Name: strings.TrimSpace(m[2])})
	}

	return devices
}
