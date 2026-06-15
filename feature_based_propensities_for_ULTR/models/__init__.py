from typing import TYPE_CHECKING

__all__ = [
    "PolicyModelBase",
    "NaivePolicyModel",
    "NaiveHOModel",
    "MaxScorePolicyModel",
    "DMPolicyModel",
    "DRPolicyModel",
    "IPSModel",
    "build_policy_model",
]


if TYPE_CHECKING:
    from feature_based_propensities_for_ULTR.models.policy_models import (
        DMPolicyModel,
        DRPolicyModel,
        IPSModel,
        MaxScorePolicyModel,
        NaiveHOModel,
        NaivePolicyModel,
        PolicyModelBase,
        build_policy_model,
    )


def __getattr__(name: str):
    if name == "build_policy_model":
        from feature_based_propensities_for_ULTR.models.policy_models import build_policy_model

        return build_policy_model

    if name in {
        "PolicyModelBase",
        "NaivePolicyModel",
        "NaiveHOModel",
        "MaxScorePolicyModel",
        "DMPolicyModel",
        "DRPolicyModel",
        "IPSModel",
    }:
        from feature_based_propensities_for_ULTR.models.policy_models import (
            DMPolicyModel,
            DRPolicyModel,
            IPSModel,
            MaxScorePolicyModel,
            NaiveHOModel,
            NaivePolicyModel,
            PolicyModelBase,
        )

        return {
            "PolicyModelBase": PolicyModelBase,
            "NaivePolicyModel": NaivePolicyModel,
            "NaiveHOModel": NaiveHOModel,
            "MaxScorePolicyModel": MaxScorePolicyModel,
            "DMPolicyModel": DMPolicyModel,
            "DRPolicyModel": DRPolicyModel,
            "IPSModel": IPSModel,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
