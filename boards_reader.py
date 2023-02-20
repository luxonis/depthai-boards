from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import marshmallow
from typing import Optional, Union, Dict, List
import copy

DEPTHAI_BOARDS_PATH = Path(__file__).parent
DEPTHAI_BOARDS_PRIVATE_PATH = Path(__file__).parent / "../depthai_boards_private" # (optional) private/custom boards should be placed in a sibling directory to this one


# Bootloader options
class BootloaderType(Enum):
	POE = 'poe' # Specifies POE bootloader
	USB = 'usb' # Specifies USB bootloader
	HEADER_USB = 'header_usb' # Specifies NOR Header Bootloader USB
	NONE = 'none' # Specifies that bootloader does not need to be flashed

	@staticmethod
	def get_default_bootloader(test_type: str):
		if 'POE' in test_type:
			return BootloaderType.POE
		elif 'FFC' in test_type:
			return BootloaderType.USB
		elif not ('LITE' in test_type or '1' in test_type):
			return BootloaderType.HEADER_USB
		else:
			return BootloaderType.NONE

@dataclass_json
@dataclass
class Options:
	bootloader: BootloaderType

	environment: Union[str, dict] = "standard" 
	""" if dict, each key represents a stage (flashing, testing, calibration) and the value 
	is the environment to use for that stage """

@dataclass_json
@dataclass
class EepromData:
	boardConf: str
	boardName: str
	boardRev: str
	productName: str
	boardCustom: str
	hardwareConf: str
	boardOptions: int
	version: int
	batchTime: int = 0 
	""" seconds since epoch """

@dataclass_json
@dataclass
class RotationType:
	r: float # roll
	p: float # pitch
	y: float # yaw

@dataclass_json
@dataclass
class TranslationType:
	x: float
	y: float
	z: float

@dataclass_json
@dataclass
class Extrinsics:
	to_cam: str
	rotation: RotationType
	specTranslation: TranslationType

@dataclass_json
@dataclass
class CameraInfo:
	name: str
	hfov: float
	type: str
	extrinsics: Optional[Extrinsics] = None
	sensor_name: str = ""
	has_autofocus: bool = False
	lens_position: int = -1

@dataclass_json
@dataclass
class StereoConfig:
	left_cam: str
	right_cam: str

@dataclass_json
@dataclass
class BoardConfig:
	cameras: Dict[str, CameraInfo]
	stereo_config: Optional[StereoConfig] = None

@dataclass_json
@dataclass
class VariantConfig:
	id: str 
	""" equivalent to the eeprom file name (inside the eeprom folder) without the extension """

	title: str
	
	description: str
	
	eeprom: str 
	""" path to eeprom file """
	
	eeprom_data: EepromData

	board_config: BoardConfig

	options: Options

@dataclass_json
@dataclass
class DeviceConfig:
	id: str 
	""" equivalent to the device file name (inside the batch folder) without the extension """

	title: str
	
	description: str

	variants: list[VariantConfig]


def update(d: Dict, u: Dict):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# construct a devices dict
# devices are represented by JSON files in depthai-boards/batch
# each device contains a list of variants which are represented by JSON files in depthai-boards/batch/eeeprom
DEVICES = []
DEVICES_TYPED: List[DeviceConfig] = []
for device_file in [*(DEPTHAI_BOARDS_PATH / "batch" ).glob("*.json"), *(DEPTHAI_BOARDS_PRIVATE_PATH / "batch" ).glob("*.json")]:
	with open(device_file, "r") as f:
		device = json.load(f)
		device["id"] = device_file.stem
		variants = device["variants"]

		# If no options are specified, use defaults
		options = {
			"bootloader": BootloaderType.get_default_bootloader(device.get("test_type", "")),
			"environment": "standard"
		}
		options.update(device.get("options", {}))
		device["options"] = options

		# Load the variants
		variants_combined = []
		for variant in variants:
			variant_combined = copy.deepcopy(device) # the variant inherits the device's properties first
			variant_combined.pop("variants") # remove the variants field from the variant
			update(variant_combined, variant) # then the variant's properties are applied on top

			# Load the eeprom data
			try: 
				eeprom_data_path = device_file.parent / variant_combined["eeprom"]
				with open(eeprom_data_path, "r") as f:
					variant_combined["eeprom_data"] = json.load(f)
					variant_combined["eeprom_file_name"] = eeprom_data_path.name
					variant_combined["id"] = eeprom_data_path.stem
			except Exception as e:
				raise Exception(f"Couldn't load eeprom file for device '{device_file.resolve()}' variant '{variant_combined.get('id', '')}'. Make sure the eeprom field is set correctly in the device file.")

			# Load the board config
			if "board_config_file" in variant_combined:
				board_config_path = device_file.parent / "../boards" / variant_combined["board_config_file"]
				if not board_config_path.exists():
					raise Exception(f"Couldn't load board config file at {board_config_path.resolve()} for device '{device_file.resolve()}'. Make sure the board_config_file field is set correctly in the device file.")
				with open(board_config_path, "r") as f:
					variant_combined["board_config"] = json.load(f).get("board_config", {}) 
			else:
				variant_combined["board_config"] = {"cameras": {}} # if no board config is specified, use an empty one (used for FCC cameras)

			# Convert the bootloader string to an enum
			variant_combined["options"]["bootloader"] = BootloaderType(options["bootloader"]) # convert string to enum

			variants_combined.append(variant_combined)
		
		device["variants"] = variants_combined

		DEVICES.append(device)

		# Convert the dict to a list of DeviceConfig objects and validate it
		try:
			device_typed: DeviceConfig = DeviceConfig.from_dict(device) # type: ignore
			DEVICES_TYPED.append(device_typed)
		except marshmallow.ValidationError as err:
			print(f"In file or in file referenced by '{device_file}': Type mismatch", err)
		except Exception as err:
			print(f"In file or in file referenced by '{device_file}': Mising field", err)



def get_device_by_id(device_id: str):
	for device in DEVICES:
		if device["id"] == device_id:
			return device
	raise KeyError(f"Device with id '{device_id}' not found")

def get_variant_by_id(variant_id: str):
	for device in DEVICES:
		for variant in device["variants"]:
			if variant["id"] == variant_id:
				return variant
	raise KeyError(f"Variant with id '{variant_id}' not found")

def get_device_by_id_typed(device_id: str):
	for device in DEVICES_TYPED:
		if device.id == device_id:
			return device
	raise KeyError(f"Device with id '{device_id}' not found")

def get_variant_by_id_typed(variant_id: str):
	for device in DEVICES_TYPED:
		for variant in device.variants:
			if variant.id == variant_id:
				return variant
	raise KeyError(f"Variant with id '{variant_id}' not found")