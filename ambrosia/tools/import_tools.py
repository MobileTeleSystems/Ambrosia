#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Do we need exceptions ?
# If package is not installed we can't pass such object to Solvers (Designer, ... )


import typing as tp

import pkg_resources


class NotInstalledPackage(Exception):
    default_message: str = "This package is not installed"

    def __init_subclass__(cls, default_message: str) -> None:
        cls.default_message = default_message
        return super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        if args:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(self.default_message, **kwargs)


class PysparkNotInstalled(NotInstalledPackage, default_message="Install pyspark or ambrosia[spark]"):
    pass


def get_installed_package_names() -> tp.List[str]:
    return [package.key for package in pkg_resources.working_set]


def check_package(package_name: str) -> bool:
    return package_name in get_installed_package_names()


def spark_installed() -> bool:
    return check_package("pyspark")


def check_spark_installed() -> tp.NoReturn:
    if not spark_installed():
        raise PysparkNotInstalled()
