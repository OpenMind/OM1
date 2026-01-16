## Install Zenoh Middleware

[Zenoh](https://zenoh.io) is a pub/sub/query protocol unifying data in motion, data at rest and computations. You need two pieces - the `eclipse-zenoh` python library and the Zenoh daemon (`zenohd`). Your Python project needs `eclipse-zenoh`, which is already added to OM1's `pyproject.toml`.

### Mac

Install the Zenoh router:

```bash
$ brew tap eclipse-zenoh/homebrew-zenoh
$ brew install zenoh
```

### Linux

Install the Zenoh router:

```bash
$ echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
$ sudo apt update
$ sudo apt install zenoh
```

## Starting/testing the Router

In a separate terminal window, start the router:

```bash
# inside OM1
zenohd -c robot_storage.json5
```

Testing:

```bash
zenohd --help
```

## Installing the Persistent Backend

See https://github.com/eclipse-zenoh/zenoh-backend-filesystem?tab=readme-ov-file#how-to-install-it

On Mac, you might need to allow `libzenoh_backend_fs.dylib` to run via `Privacy and Security`. Just try to run it separately - e.g. via the terminal and then `approve` the various popup messages. Once you have run `libzenoh_backend_fs.dylib` once, it will be cached and then `zenoh` can find it the next time. This is good, but can be confusing if you are trying to upgrade `libzenoh_backend_fs.dylib` but it keeps using an older cached version.

The RocksDB database is where ever you set the path to:

```bash
export ZENOH_BACKEND_FS_ROOT=$PWD/zenohdb/
```

The `RocksDB version: 9.9.3` is set up at `ZENOH_BACKEND_FS_ROOT`, where it creates a system of folders. For example, a PUT to `/robot/audio` creates a file at `/zenohdb/robot/audio`, which then contains the most recent value of the `robot/audio` key.

## Using the REST API at ::9500

You can use `curl` to publish and query the **latest** keys/values:

```bash
# Put values that will be stored under ${ZENOH_BACKEND_FS_ROOT}/robot
curl -X PUT -d "HELLO WORLD" http://localhost:9500/robot
curl -X PUT -d "HELLO WORLD A" http://localhost:9500/robot/audio

# Retrieve the values
curl http://localhost:9500/robot
curl http://localhost:9500/robot/audio
```

To be clear, the system only saves the most recent key/value pair.
