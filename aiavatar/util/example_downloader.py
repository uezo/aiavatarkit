import io
import logging
import os
import tarfile
import urllib.request

logger = logging.getLogger(__name__)


def download_example(example_path: str, branch: str = "main") -> str:
    """Download example files from the aiavatarkit GitHub repository.

    Args:
        example_path: Path relative to examples/ directory (e.g. "websocket/html").
        branch: Git branch to download from. Defaults to "main".

    Returns:
        The local directory path where files were extracted.
    """
    local_dir = example_path.split("/")[-1]

    if os.path.isdir(local_dir):
        logger.info(f"Directory '{local_dir}' already exists, skipping download.")
        return local_dir

    tarball_url = f"https://github.com/uezo/aiavatarkit/archive/refs/heads/{branch}.tar.gz"
    tarball_prefix = f"aiavatarkit-{branch}"

    # Build prefixes for filtering and stripping tar members
    target_prefix = f"{tarball_prefix}/examples/{example_path}"
    parts = example_path.split("/")
    if len(parts) > 1:
        strip_prefix = f"{tarball_prefix}/examples/{'/'.join(parts[:-1])}/"
    else:
        strip_prefix = f"{tarball_prefix}/examples/"

    logger.info(f"Downloading {example_path} from GitHub ({branch} branch)...")

    with urllib.request.urlopen(tarball_url) as response:
        data = response.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if member.name == target_prefix or member.name.startswith(target_prefix + "/"):
                member.name = member.name[len(strip_prefix):]
                members.append(member)

        if not members:
            raise FileNotFoundError(
                f"Example '{example_path}' not found in repository."
            )

        try:
            tar.extractall(members=members, filter="data")
        except TypeError:
            tar.extractall(members=members)

    logger.info(f"Downloaded to '{local_dir}'")

    return local_dir
