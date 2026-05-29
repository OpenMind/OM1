package zenoh

/*
#cgo CFLAGS: -I${SRCDIR}/../../.zenoh-c/include
#cgo LDFLAGS: -L${SRCDIR}/../../.zenoh-c/lib -lzenohc
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../../.zenoh-c/lib

#include "zenoh.h"
#include <stdlib.h>

static z_result_t om1_zenoh_open(z_owned_session_t *session, const char *endpoint_json) {
	z_owned_config_t config;
	if (z_config_default(&config) != Z_OK) {
		return -1;
	}
	if (endpoint_json != NULL) {
		zc_config_insert_json5(z_config_loan_mut(&config), "mode", "\"client\"");
		zc_config_insert_json5(z_config_loan_mut(&config), "connect/endpoints", endpoint_json);
	}
	return z_open(session, z_config_move(&config), NULL);
}

static z_result_t om1_zenoh_put(z_owned_session_t *session, const char *key,
                                const uint8_t *data, size_t len) {
	z_owned_keyexpr_t keyexpr;
	if (z_keyexpr_from_str(&keyexpr, key) != Z_OK) {
		return -1;
	}
	z_owned_bytes_t payload;
	z_bytes_copy_from_buf(&payload, data, len);
	z_put_options_t opts;
	z_put_options_default(&opts);
	z_result_t res = z_put(z_session_loan(session), z_keyexpr_loan(&keyexpr),
	                       z_bytes_move(&payload), &opts);
	z_keyexpr_drop(z_keyexpr_move(&keyexpr));
	return res;
}

static void om1_zenoh_close(z_owned_session_t *session) {
	z_close(z_session_loan_mut(session), NULL);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Session wraps a zenoh-c session.
type Session struct {
	owned C.z_owned_session_t
}

// Open opens a zenoh session. If endpoint is non-empty (e.g. "tcp/127.0.0.1:7447"),
// the session connects as a client to that router; otherwise peer/discovery mode is used.
func Open(endpoint string) (*Session, error) {
	s := &Session{}
	var cEndpoint *C.char
	if endpoint != "" {
		ep := fmt.Sprintf(`["%s"]`, endpoint)
		cEndpoint = C.CString(ep)
		defer C.free(unsafe.Pointer(cEndpoint))
	}
	if rc := C.om1_zenoh_open(&s.owned, cEndpoint); rc != 0 {
		return nil, fmt.Errorf("zenoh: open failed (rc=%d)", int(rc))
	}
	return s, nil
}

// Put publishes raw bytes to the given key expression.
func (s *Session) Put(key string, data []byte) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var ptr *C.uint8_t
	if len(data) > 0 {
		ptr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}

	rc := C.om1_zenoh_put(&s.owned, cKey, ptr, C.size_t(len(data)))
	if rc != 0 {
		return fmt.Errorf("zenoh: put failed (rc=%d)", int(rc))
	}
	return nil
}

// Close closes the session.
func (s *Session) Close() {
	C.om1_zenoh_close(&s.owned)
}
