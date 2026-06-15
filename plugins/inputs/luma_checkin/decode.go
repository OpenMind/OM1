package luma_checkin

import (
	"bytes"
	"errors"
	"image"
	_ "image/jpeg"

	"github.com/makiuchi-d/gozxing"
	"github.com/makiuchi-d/gozxing/qrcode"
)

// errQRNotFound signals that no QR code was found in the frame. Callers should
// treat this as the common case and skip silently.
var errQRNotFound = errors.New("luma_checkin: no qr code in frame")

// decodeQR decodes a single QR code from a JPEG-encoded frame. It returns the
// raw text payload of the code, or errQRNotFound if no code is present.
func decodeQR(jpegBytes []byte) (string, error) {
	img, _, err := image.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		return "", err
	}
	bmp, err := gozxing.NewBinaryBitmapFromImage(img)
	if err != nil {
		return "", err
	}
	reader := qrcode.NewQRCodeReader()
	result, err := reader.Decode(bmp, nil)
	if err != nil {
		return "", errQRNotFound
	}
	return result.GetText(), nil
}
