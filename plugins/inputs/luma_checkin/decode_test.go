package luma_checkin

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"testing"

	"github.com/makiuchi-d/gozxing"
	"github.com/makiuchi-d/gozxing/qrcode"
)

func TestDecodeQRRoundTrip(t *testing.T) {
	const payload = "https://luma.com/check-in/evt-test?pk=g-test-123"

	jpegBytes := encodeQRAsJPEG(t, payload, 360)

	got, err := decodeQR(jpegBytes)
	if err != nil {
		t.Fatalf("DecodeQR: %v", err)
	}
	if got != payload {
		t.Errorf("payload: got %q want %q", got, payload)
	}
}

func TestDecodeQRReturnsNotFoundOnBlankFrame(t *testing.T) {
	jpegBytes := encodeBlankJPEG(t, 100, 100)

	_, err := decodeQR(jpegBytes)
	if !errors.Is(err, errQRNotFound) {
		t.Fatalf("expected ErrQRNotFound, got %v", err)
	}
}

func encodeQRAsJPEG(t *testing.T, payload string, size int) []byte {
	t.Helper()
	writer := qrcode.NewQRCodeWriter()
	bm, err := writer.Encode(payload, gozxing.BarcodeFormat_QR_CODE, size, size, nil)
	if err != nil {
		t.Fatalf("encode QR: %v", err)
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, bm, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return buf.Bytes()
}

func encodeBlankJPEG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := newBlankImage(w, h)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return buf.Bytes()
}

func newBlankImage(w, h int) image.Image {
	img := image.NewGray(image.Rect(0, 0, w, h))
	white := color.Gray{Y: 255}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, white)
		}
	}
	return img
}
