package memory

import "go.uber.org/zap"

// NewManager creates a MemoryManager based on the cloudConnection flag.
func NewManager(memoryRoot, apiKey string, cloudConnection bool, log *zap.Logger) MemoryManager {
	if cloudConnection {
		return NewCloudManager(memoryRoot, apiKey, log)
	}
	return NewLocalManager(memoryRoot, apiKey, log)
}
