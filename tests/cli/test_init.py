import aiavatar.cli.components as cli_components


def test_build_components_is_available_from_cli_package():
    from aiavatar.cli import build_components

    assert build_components is cli_components.build_components
