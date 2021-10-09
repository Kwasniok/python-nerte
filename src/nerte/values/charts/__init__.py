"""Module for charts embedding representations of manifolds into R^3."""

from .chart_1_to_3 import Chart1DTo3D, CanonicalImmersionChart1DTo3D
from .chart_2_to_3 import Chart2DTo3D, CanonicalImmersionChart2DTo3D
from .chart_3_to_3 import Chart3DTo3D, IdentityChart3D

# charts interplaying with transformations
from .domain_transformed_chart_3_to_3 import DomainTransformedChart3DTo3D
from .codomain_transformed_chart_3_to_3 import CodomainTransformedChart3DTo3D
from .codomain_transformed_chart_2_to_3 import CodomainTransformedChart2DTo3D
