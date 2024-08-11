from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer

from spd.utils import to_root_path

# This is a type for pydantic configs that will convert all relative paths
# to be relative to the ROOT_DIR of e2e_sae
RootPath = Annotated[Path, BeforeValidator(to_root_path), PlainSerializer(lambda x: str(x))]

TrigParams = tuple[float, float, float, float, float, float, float]
