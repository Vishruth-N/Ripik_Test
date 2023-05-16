"""
 @author: HimanshuMittal01
 @organization: ripiktech
"""

import pandas as pd
from typing import Tuple, List, Dict, Any, Union

from optimus.configure import setup_config_params, setup_logging
from optimus.common.exceptions import SequenceError
from optimus.loaders import PandasFilePath, get_loader
from optimus.schedule_full import schedule_full
from optimus.utils.views import (
    view_by_commit,
    view_by_machine,
    view_by_product,
    view_by_quantity,
    view_by_production,
)
from optimus.insights.insights import obtain_insights
from optimus.mrp import MRP
from optimus.insights.capacity import room_capacity


class Optimus:
    def __init__(self, user_parameters: Dict[str, Any] = {}) -> None:
        # Setup logging
        setup_logging()

        # Setup configuration paramteters
        self.config = setup_config_params(user_parameters=user_parameters)

        # Init vars
        self._reset_state()

    def _reset_state(self) -> None:
        # Clear any loaded data
        self.data = None

        # No schedule created
        self._is_schedule_created = False

        # Clear cached output
        self._output = {
            "machine_view": None,
            "product_view": None,
            "quantity_view": None,
            "production_view": None,
            "commit_view": None,
            "insights": None,
            "mrp": None,
            "room_insights": None,
        }

    def create_schedule(
        self, data_files: Dict[str, PandasFilePath]
    ) -> Tuple[bool, str]:
        # Reset
        self._reset_state()

        # Get loader based on the client ID
        loader = get_loader(config=self.config)

        # Load data
        self.data = loader.load_all(data_files=data_files)

        # Delete loader object to save memory
        del loader

        # Optimize
        (
            self.products,
            self.machines,
            self.normalised_demand,
            self.dropped_reasons,
            self.feasible_batches,
        ) = schedule_full(
            data=self.data,
            config=self.config,
            inner_optimisation_params=dict(
                num_iterations=self.config["inner_num_iterations"]
            ),
        )

        if len(self.feasible_batches) == 0:
            return False, "Feasible commit is empty"

        # Schedule is successfully created
        self._is_schedule_created = True
        return True, "Successfully created the schedule"

    def get_machine_view(self) -> pd.DataFrame:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["machine_view"] is None:
            self._output["machine_view"] = view_by_machine(
                machines=self.machines,
                execution_start=self.config["execution_start"],
            )
        return self._output["machine_view"]

    def get_product_view(self) -> pd.DataFrame:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["product_view"] is None:
            self._output["product_view"] = view_by_product(
                machines=self.machines, execution_start=self.config["execution_start"]
            )
        return self._output["product_view"]

    def get_quantity_view(self) -> pd.DataFrame:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["quantity_view"] is None:
            self._output["quantity_view"] = view_by_quantity(
                machines=self.machines,
                execution_start=self.config["execution_start"],
            )
        return self._output["quantity_view"]

    def get_production_view(self) -> pd.DataFrame:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["production_view"] is None:
            self._output["production_view"] = view_by_production(
                machines=self.machines, config=self.config
            )
        return self._output["production_view"]

    def get_commit_view(self) -> pd.DataFrame:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["commit_view"] is None:
            self._output["commit_view"] = view_by_commit(
                machines=self.machines,
                actual_demand=self.data["df_forecasted_demand"],
                normalised_demand=self.normalised_demand,
                feasible_batches=self.feasible_batches,
                dropped_reasons=self.dropped_reasons,
                config=self.config,
            )
        return self._output["commit_view"]

    def create_insights(self) -> Dict[str, Any]:
        # Obtain required views if not in cache
        if self._output["insights"] is None:
            machine_view = self._output["machine_view"]
            if machine_view is None:
                machine_view = self.get_machine_view()

            product_view = self._output["product_view"]
            if product_view is None:
                product_view = self.get_product_view()

            production_view = self._output["production_view"]
            if production_view is None:
                production_view = self.get_production_view()

            commit_view = self._output["commit_view"]
            if commit_view is None:
                commit_view = self.get_commit_view()

            # Generate insights
            self._output["insights"] = obtain_insights(
                machine_view=machine_view,
                product_view=product_view,
                production_view=production_view,
                commit_view=commit_view,
                products=self.products,
                config=self.config,
            )

        return self._output["insights"]

    def create_room_insights(self) -> Dict[str, Any]:
        if not self._is_schedule_created:
            raise SequenceError("Schedule is not created so can't produce the view")

        if self._output["room_insights"] is None:
            self._output["room_insights"] = room_capacity(machines=self.machines)
        return self._output["room_insights"]

    def create_mrp(
        self, sustaining_days: int = 90
    ) -> Dict[str, List[Union[str, float]]]:
        # Get product view
        if self._output["product_view"] is None:
            self._output["product_view"] = view_by_product(
                machines=self.machines, execution_start=self.config["execution_start"]
            )

        mrp = MRP(
            inventory=self.data["df_inventory"].reset_index(drop=True),
            products=self.products,
            product_view=self._output["product_view"],
            execution_start=self.config["execution_start"],
            df_phantom_items=self.data["df_phantom_items"],
        )
        mrp_output = mrp.get_pr_all(sustaining_days=sustaining_days)

        return mrp_output
