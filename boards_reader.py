from pathlib import Path
import json
from enum import Enum
from pydantic import BaseModel, ValidationError
from typing import Optional, Union, Dict, List, Tuple
import copy

DEPTHAI_BOARDS_PATH = Path(__file__).parent
DEPTHAI_BOARDS_PRIVATE_PATH = Path(__file__).parent / "../depthai_boards_private" # (optional) private/custom boards should be placed in a sibling directory to this one


# Bootloader options
class BootloaderType(str, Enum):
	POE = 'poe' # Specifies POE bootloader
	USB = 'usb' # Specifies USB bootloader
	HEADER_USB = 'header_usb' # Specifies NOR Header Bootloader USB
	NONE = 'none' # Specifies that bootloader does not need to be flashed
	TEST = 'test' # Only test reads & writes the bootloader

	@staticmethod
	def get_default_bootloader(test_type: str):
		if 'POE' in test_type:
			return BootloaderType.POE
		elif not ('LITE' in test_type or '1' in test_type):
			return BootloaderType.HEADER_USB
		else:
			return BootloaderType.NONE


class CameraSettings(BaseModel):
	sharpness: Optional[int] = None
	luma_denoise: Optional[int] = None
	chroma_denoise: Optional[int] = None
	exposure: Optional[Tuple[int, int]] = None
	""" Tuple of (exposure_time, ISO_gain) values. Exposure is in microseconds."""
	lens_position: Optional[int] = None

	isp_scale: Optional[Tuple[int, int]] = None
	""" Tuple of (numerator, denominator) values. The image is scaled by
	numerator/denominator. Only applicable to color cameras. """

class TVCalibrationSettings(BaseModel):
	camera_settings: Dict[str, CameraSettings] = {}
	""" Dictionary with camera sockets as keys and CameraSettings as values. Used to set
	camera settings (sharpness, exposure, ...) for TV calibration. """

	n_charuco_markers_per_row: int = 19
	""" The number of charuco markers per row in the calibration pattern. The size of a
	charuco square is determined by dividing the width of the TV by this number. """


class BasicCameraInfo(BaseModel):
	name: str = ""
	socket: str
	type: str # color, mono, tof

	def __hash__(self) -> int:
		return hash((self.name, self.socket, self.type))

	def dict(self, *args, **kwargs):
		return f"{self.name} ({self.socket})"


# Base model mirror of dai.DeviceBootlodar.Config
class BootloaderConfig(BaseModel):
	@staticmethod
	def str_to_ip(ip: str) -> int:
		""" Converts an IPv4 string to an integer."""
		return sum([int(x) << (8 * i) for i, x in enumerate(ip.split("."))])
	

	class NetworkConfig(BaseModel):
		ipv4: int = 0
		ipv4Dns: int = 0
		ipv4DnsAlt: int = 0
		ipv4Gateway: int = 0
		ipv4Mask: int = 0
		ipv6: List[int] = [0, 0, 0, 0]
		ipv6Dns: List[int] = [0, 0, 0, 0]
		ipv6DnsAlt: List[int] = [0, 0, 0, 0]
		ipv6Gateway: List[int] = [0, 0, 0, 0]
		ipv6Prefix: int = 0
		mac: List[int] = [0, 0, 0, 0, 0, 0]
		staticIpv4: bool = False
		staticIpv6: bool = False
		timeoutMs: int = 30000

		def __init__(self, **data):
			if "ipv4" in data and isinstance(data["ipv4"], str):
				data["ipv4"] = BootloaderConfig.str_to_ip(data["ipv4"])
			if "ipv4Dns" in data and isinstance(data["ipv4Dns"], str):
				data["ipv4Dns"] = BootloaderConfig.str_to_ip(data["ipv4Dns"])
			if "ipv4DnsAlt" in data and isinstance(data["ipv4DnsAlt"], str):
				data["ipv4DnsAlt"] = BootloaderConfig.str_to_ip(data["ipv4DnsAlt"])
			if "ipv4Gateway" in data and isinstance(data["ipv4Gateway"], str):
				data["ipv4Gateway"] = BootloaderConfig.str_to_ip(data["ipv4Gateway"])
			if "ipv4Mask" in data and isinstance(data["ipv4Mask"], str):
				data["ipv4Mask"] = BootloaderConfig.str_to_ip(data["ipv4Mask"])
			if "ipv6" in data and isinstance(data["ipv6"], str):
				data["ipv6"] = [int(x, 16) for x in data["ipv6"].split(":")]
			if "ipv6Dns" in data and isinstance(data["ipv6Dns"], str):
				data["ipv6Dns"] = [int(x, 16) for x in data["ipv6Dns"].split(":")]
			if "ipv6DnsAlt" in data and isinstance(data["ipv6DnsAlt"], str):
				data["ipv6DnsAlt"] = [int(x, 16) for x in data["ipv6DnsAlt"].split(":")]
			if "ipv6Gateway" in data and isinstance(data["ipv6Gateway"], str):
				data["ipv6Gateway"] = [int(x, 16) for x in data["ipv6Gateway"].split(":")]
			if "mac" in data and isinstance(data["mac"], str):
				data["mac"] = [int(x, 16) for x in data["mac"].split(":")]
			super().__init__(**data)
    
	class UsbConfig(BaseModel):
		maxUsbSpeed: int = 3
		pid: int = 0xF63C
		timeoutMs: int = 3000
		vid: int = 0x03E7

	network: NetworkConfig = NetworkConfig()
	usb: UsbConfig = UsbConfig()

class MacAddressGeneratingMethod(str, Enum):
	last_serial_number_digits = "last_serial_number_digits"
	database_sourced = "database_sourced"

class MacAddressConfig(BaseModel):
	prefix: str
	""" MAC address prefix to be used for all generated MAC addresses. """

	prefix_bits: int
	""" Number of bits to be used from the prefix. """

class MacAddressSerialBasedConfig(MacAddressConfig):
	serial_number_digits: int
	""" Number of digits to be used from the serial number. """
	
class MacAddressDBBasedConfig(MacAddressConfig):
	region_id: str
	""" MAC region ID in database to retrieve from. """
	
class FlashMacAddressConfig(BaseModel):
	generating_method: MacAddressGeneratingMethod
	""" Method used to generate the MAC address. """

	config: Union[MacAddressSerialBasedConfig, MacAddressDBBasedConfig]
	""" Config for corresponding MAC generation method used. """

class Options(BaseModel):
	bootloader: BootloaderType
	bootloader_config: Optional[BootloaderConfig] = None

	environment: Union[str, dict] = "standard"
	""" if dict, each key represents a stage (flashing, testing, calibration) and the value
	is the environment to use for that stage """

	has_serial_code: bool = False
	"""Does the device have a serial code?"""

	imu: bool = True
	""" Does the board have an IMU or not? """

	imu_kind: Optional[str] = None
	""" The tests will check for this specific IMU kind if it's specified. """

	usb3: bool = True
	"""Does the board support USB3?"""

	jpeg: bool = True
	"""Does the board support JPEG encoding? (or is it tested?)"""

	eth: bool = False
	"""Does the board support Ethernet? (Is it tested?)"""

	nor_size: int = 0
	"""Specify the size of the NOR flash in bytes."""

	eeprom: bool = True
	"""Should the eeprom be flashed?"""

	emmc_size: int = 0
	"""Specify the size of the eMMC storage size in bytes (if the device has it)."""

	websocket_capture: bool = False
	""" This should be set to 'True' for cameras (e.g. OAK-D-CM4) that don't work with depthai
	library directly and need a websocket server to stream the images to the
	calibration node.  """

	tv_calibration: TVCalibrationSettings = TVCalibrationSettings()
	""" Settings for TV calibration. """

	platform: str = ""
	""" Specify platform of device. """

	skip_eeprom_check: bool = False

	cameras: List[BasicCameraInfo] = []
	"""List of cameras on board. (If specified this camera config is preferred over board_options for testing)"""

	flash_mac_address: Optional[FlashMacAddressConfig] = None
	""" Configuration for generating MAC addresses during flashing. """

	ssh_password: str = ""
	""" Password for SSH connection to the device. """

class EepromData(BaseModel):
	boardConf: Optional[str] = None
	boardName: Optional[str] = None
	boardRev: Optional[str] = None
	productName: Optional[str] = None
	boardCustom: Optional[str] = None
	hardwareConf: Optional[str] = None
	boardOptions: Optional[int] = None
	version: Optional[int] = None
	batchTime: int = 0
	""" seconds since epoch """


class RotationType(BaseModel):
	r: float # roll
	p: float # pitch
	y: float # yaw

class TranslationType(BaseModel):
	x: float
	y: float
	z: float

class Extrinsics(BaseModel):
	to_cam: str
	rotation: RotationType
	specTranslation: TranslationType

class CameraInfo(BaseModel):
	name: str = ""
	hfov: float = 0.0
	type: str = ""
	camera_model: str = "perspective"
	calib_model: str = "perspective_NORMAL"
	""" Camera model can be either 'perspective' or 'fisheye'. """
	extrinsics: Optional[Extrinsics] = None
	sensor_name: str = ""
	has_autofocus: bool = False
	lens_position: int = -1

class ImuSensorInfo(BaseModel):
	name: str = ""
	extrinsics: Optional[Extrinsics] = None

class StereoConfig(BaseModel):
	left_cam: str
	right_cam: str

class ImuExtrinsics(BaseModel):
	sensors: Dict[str, ImuSensorInfo]

class BoardConfig(BaseModel):
	cameras: Dict[str, CameraInfo]
	stereo_config: Optional[StereoConfig] = None
	imuExtrinsics: Optional[ImuExtrinsics] = None

class VariantConfig(BaseModel):
	id: str
	""" equivalent to the eeprom file name (inside the eeprom folder) without the extension """

	title: str

	description: str

	eeprom: str
	""" path to eeprom file """

	eeprom_data: EepromData

	board_config: BoardConfig

	board_config_2: BoardConfig

	options: Options

	fip: Optional[str] = None
	""" Name of the FIP fipe to be flashed."""

	cdt: Optional[str] = None
	"""Name of the CDT file to be flashed. (RVC4 only)"""

	os: Optional[str] = None
	"""Name of the OS zip to be flashed."""

	configs: Optional[List[str]] = None
	""" List of config tars names to be flashed. """

	test_suite: str = ""
	""" Specify which test_suite to use. """

	test_station_config: Optional[str] = None
	"""Path of the test_station_config, look at stage_testing/test_station/config/__init__.py for more info."""


class DeviceConfig(BaseModel):
	id: str
	""" equivalent to the device file name (inside the batch folder) without the extension """

	title: str

	description: str

	os: Optional[str] = None
	"""The common OS which to use on all of this devices' variants, unless explicitly overwritten by the VariantConfig"""

	variants: List[VariantConfig]


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
	try:
		with open(device_file, "r") as f:
			device = json.load(f)
	except json.decoder.JSONDecodeError as e:
		raise Exception(f"Couldn't parse device file at {device_file.resolve()}. Make sure the file is valid JSON. \n{e}")
	except Exception as e:
		raise Exception(f"Couldn't load device file at {device_file.resolve()}. Make sure the file exists.")

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
		except json.decoder.JSONDecodeError as e:
			raise Exception(f"Couldn't parse eeprom file for device '{device_file.resolve()}' variant '{variant_combined.get('id', '')}'. Make sure the eeprom file is valid JSON. \n{e}")
		except Exception as e:
			raise Exception(f"Couldn't load eeprom file for device '{device_file.resolve()}' variant '{variant_combined.get('id', '')}'. Make sure the eeprom field is set correctly in the device file.")

		# Load the board config
		if "board_config_file" in variant_combined:

			if type(variant_combined["board_config_file"]) is list:
				board_config_path=[]
				for index, board in enumerate(variant_combined["board_config_file"]):
					board_config_path.append(device_file.parent / "../boards" / variant_combined["board_config_file"][index])
					try:
						with open(board_config_path[index], "r") as f:
							variant_combined[f"board_config_{index}"]=(json.load(f).get("board_config", {}))
					except json.decoder.JSONDecodeError as e:
						raise Exception(f"Couldn't parse board config file at {board_config_path[index].resolve()} for device '{device_file.resolve()}'. Make sure the board config file is valid JSON. \n{e}")
					except Exception as e:
						raise Exception(f"Couldn't load board config file at {board_config_path[index].resolve()} for device '{device_file.resolve()}'. Make sure the board_config_file field is set correctly in the device file.")
			else:
				board_config_path = device_file.parent / "../boards" / variant_combined["board_config_file"]
				try:
					with open(board_config_path, "r") as f:
						variant_combined["board_config"] = json.load(f).get("board_config", {})
				except json.decoder.JSONDecodeError as e:
					raise Exception(f"Couldn't parse board config file at {board_config_path.resolve()} for device '{device_file.resolve()}'. Make sure the board config file is valid JSON. \n{e}")
				except Exception as e:
					raise Exception(f"Couldn't load board config file at {board_config_path.resolve()} for device '{device_file.resolve()}'. Make sure the board_config_file field is set correctly in the device file.")
		else:
			variant_combined["board_config"] = {"cameras": {}} # if no board config is specified, use an empty one (used for FCC cameras)
		# Convert the bootloader string to an enum
		variant_combined["options"]["bootloader"] = BootloaderType(variant_combined["options"]["bootloader"]) # convert string to enum

		if "board_config_file_2" in variant_combined:
			board_config_path = device_file.parent / "../boards" / variant_combined["board_config_file_2"]
			try:
				with open(board_config_path, "r") as f:
					variant_combined["board_config_2"] = json.load(f).get("board_config", {})
			except json.decoder.JSONDecodeError as e:
					raise Exception(f"Couldn't parse board config file at {board_config_path.resolve()} for device '{device_file.resolve()}'. Make sure the board config file is valid JSON. \n{e}")
			except Exception as e:
					raise Exception(f"Couldn't load board config file at {board_config_path.resolve()} for device '{device_file.resolve()}'. Make sure the board_config_file field is set correctly in the device file.")
		else:
			variant_combined["board_config_2"] = {"cameras": {}} # if no board config is specified, use an empty one (used for FCC cameras)

		variants_combined.append(variant_combined)

	device["variants"] = variants_combined

	DEVICES.append(device)

	# Convert the dict to a list of DeviceConfig objects and validate it
	try:
		device_typed: DeviceConfig = DeviceConfig(**device) # type: ignore
		DEVICES_TYPED.append(device_typed)
	except ValidationError as err:
		raise Exception(f"Validation error in file or in file referenced by '{device_file}':\n {err}\n")



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

def get_variant_by_eeprom_typed(calibration):
	eeprom = calibration.getEepromData()

	for device in DEVICES_TYPED:
		for variant in device.variants:
			if (eeprom.productName.upper().replace(' ', '-') == variant.eeprom_data.productName.upper().replace(' ', '-') and
				eeprom.boardName == variant.eeprom_data.boardName and
				eeprom.boardRev == variant.eeprom_data.boardRev and
				eeprom.hardwareConf == variant.eeprom_data.hardwareConf and
				eeprom.boardConf == variant.eeprom_data.boardConf):
				return variant
	raise KeyError(
		f"Variant with eeprom data 'productName: {eeprom.productName}, \
			boardName: {eeprom.boardName}, boardRev: {eeprom.boardRev}, \
			boardConf: {eeprom.boardConf} \
				hardwareConf: {eeprom.hardwareConf}' not found")
