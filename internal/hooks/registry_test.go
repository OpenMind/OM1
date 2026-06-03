package hooks

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRegisterAndLookupHook(t *testing.T) {
	fn := func(*Runner, context.Context, map[string]any, map[string]any) error { return nil }
	RegisterHook("mod", "lookup_"+t.Name(), fn)

	got, ok := lookupHook("mod", "lookup_"+t.Name())
	require.True(t, ok)
	require.NotNil(t, got)

	_, ok = lookupHook("mod", "never_registered")
	require.False(t, ok)
}

func TestRegisterHookDuplicatePanics(t *testing.T) {
	fn := func(*Runner, context.Context, map[string]any, map[string]any) error { return nil }
	RegisterHook("dup", "fn_"+t.Name(), fn)
	require.PanicsWithValue(t,
		"hooks: duplicate function hook registration for dup.fn_"+t.Name(),
		func() { RegisterHook("dup", "fn_"+t.Name(), fn) },
	)
}
