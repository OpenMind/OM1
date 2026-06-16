package config

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/stretchr/testify/require"
)

func compileSchema(t *testing.T, path string) *jsonschema.Schema {
	t.Helper()

	raw, err := os.ReadFile(path)
	require.NoError(t, err, "read schema %s", path)

	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	require.NoError(t, err, "parse schema %s", path)

	c := jsonschema.NewCompiler()
	require.NoError(t, c.AddResource(filepath.Base(path), doc), "add schema %s", path)

	sch, err := c.Compile(filepath.Base(path))
	require.NoError(t, err, "compile schema %s", path)

	return sch
}

func isModeConfig(data map[string]any) bool {
	_, hasModes := data["modes"]
	_, hasDefault := data["default_mode"]
	return hasModes && hasDefault
}

func TestConfigFilesMatchSchema(t *testing.T) {
	configDir := filepath.Join("..", "..", "config")
	schemaDir := filepath.Join(configDir, "schema")

	singleSchema := compileSchema(t, filepath.Join(schemaDir, "single_mode_schema.json"))
	multiSchema := compileSchema(t, filepath.Join(schemaDir, "multi_mode_schema.json"))

	entries, err := os.ReadDir(configDir)
	require.NoError(t, err, "read config dir")

	var configFiles []string
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json5" {
			continue
		}
		configFiles = append(configFiles, entry.Name())
	}
	require.NotEmpty(t, configFiles, "expected at least one .json5 config file")

	for _, name := range configFiles {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join(configDir, name))
			require.NoError(t, err, "read config")

			jsonData, err := stripJSON5([]byte(expandEnv(string(raw))))
			require.NoError(t, err, "convert JSON5 to JSON")

			inst, err := jsonschema.UnmarshalJSON(bytes.NewReader(jsonData))
			require.NoError(t, err, "parse config as JSON")

			data, ok := inst.(map[string]any)
			require.True(t, ok, "config root must be a JSON object")

			schema := singleSchema
			if isModeConfig(data) {
				schema = multiSchema
			}

			if err := schema.Validate(inst); err != nil {
				t.Fatalf("%s failed schema validation:\n%v", name, err)
			}
		})
	}
}
