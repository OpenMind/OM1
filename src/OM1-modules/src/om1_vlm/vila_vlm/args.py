import argparse


class VILAArgParser(argparse.ArgumentParser):
    """
    Argument parser for VILA VLM configuration.
    """

    Defaults = ["vila", "standalone"]  #: The default options for VILA configuration

    def __init__(self, extras=Defaults, **kwargs):
        """
        Initialize the VILA argument parser with model-specific options.
        """
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )

        # VILA configuration
        if "vila" in extras:
            self.add_argument(
                "--vila-host",
                type=str,
                default="localhost",
                help="VILA server hostname",
            )
            self.add_argument(
                "--vila-port", type=int, default=8000, help="VILA server WebSocket port"
            )
            self.add_argument(
                "--vila-batch-size",
                type=int,
                default=5,
                help="Number of frames to process in a batch",
            )
            self.add_argument(
                "--vila-model-path",
                type=str,
                default="/app/Efficient-Large-Model/VILA1.5-3B",
                help="Path to the VILA model",
            )

        # Standalone mode options
        if "standalone" in extras:
            self.add_argument(
                "--fps",
                type=int,
                default=10,
                help="Frames per second for video processing",
            )

    def parse_args(self, **kwargs):
        """
        Parse command-line arguments with additional configuration.
        """
        args = super().parse_args(**kwargs)
        return args
