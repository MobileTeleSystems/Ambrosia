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

"""
Ambrosia
===============================

Ambrosia is a Python library for A/B tests design, split and effect
measurement. It provides rich set of methods for conducting full
A/B test pipeline. In particular, a design stage could be performed
using data from both pandas and spark dataframes with either
theoretical or empirical approach. Split methods support different
strategies and multigroup split. Final effect measurement stage could
be gently conducted via testing tools that allow to measure relative
and absolute effects and construct corresponding confidence intervals
for continious and binary variables. Testing tools as well as design
support significant number of statistical criteria, like t-test,
non-parametric ones, and bootstrap. For additional A/B tests support
package provides features and tools for data preproccesing and
experiment acceleration.

See "https://ambrosia.readthedocs.io" for complete documentation.

Subpackages
------------
    preprocessing - Experiment data preprocessing
    designer - Experiments design
    splitter - Groups split
    tester - Effects measurement
    tools - Core methods
    spark_tools - Spark methods
"""

from ambrosia.version import __version__
