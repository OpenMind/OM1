# src/cli/api_keys.py

"""
CLI tool for managing API keys

Usage:
    python -m src.cli.api_keys set openmind
    python -m src.cli.api_keys get openmind
    python -m src.cli.api_keys delete openmind
    python -m src.cli.api_keys list
"""

import sys
import argparse
from getpass import getpass
from src.utils.api_key_manager import APIKeyManager, mask_api_key

def main():
    parser = argparse.ArgumentParser(description='Manage OM1 API keys securely')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set API key')
    set_parser.add_argument('provider', help='Provider name (e.g., openmind, openai)')
    set_parser.add_argument('--key', help='API key (will prompt if not provided)')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get API key')
    get_parser.add_argument('provider', help='Provider name')
    
    # Delete command
    del_parser = subparsers.add_parser('delete', help='Delete API key')
    del_parser.add_argument('provider', help='Provider name')
    
    # List command
    subparsers.add_parser('list', help='List configured providers')
    
    args = parser.parse_args()
    manager = APIKeyManager()
    
    if args.command == 'set':
        api_key = args.key or getpass(f"Enter API key for {args.provider}: ")
        if manager.set_api_key(args.provider, api_key):
            print(f"✅ API key set for {args.provider}")
        else:
            print(f"❌ Failed to set API key")
            sys.exit(1)
    
    elif args.command == 'get':
        api_key = manager.get_api_key(args.provider)
        if api_key:
            print(f"{args.provider}: {mask_api_key(api_key)}")
        else:
            print(f"❌ No API key found for {args.provider}")
            sys.exit(1)
    
    elif args.command == 'delete':
        if manager.delete_api_key(args.provider):
            print(f"✅ API key deleted for {args.provider}")
        else:
            print(f"❌ Failed to delete API key")
    
    elif args.command == 'list':
        providers = ['openmind', 'openai', 'anthropic', 'gemini', 'deepseek']
        print("Configured API keys:")
        for provider in providers:
            key = manager.get_api_key(provider)
            if key:
                print(f"  ✓ {provider}: {mask_api_key(key)}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
