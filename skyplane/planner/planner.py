from importlib.resources import path

from skyplane import compute
from skyplane.planner.solver import ThroughputProblem
from skyplane.planner.topology import ReplicationTopology
from skyplane.planner.gateway_program import GatewayProgram


class Planner:
    def __init__(self, src_provider: str, src_region, dst_provider: str, dst_region: str):
        self.src_provider = src_provider
        self.src_region = src_region
        self.dst_provider = dst_provider
        self.dst_region = dst_region

    def plan(self) -> GatewayProgram:
        raise NotImplementedError


class DirectPlanner(Planner):
    def __init__(self, src_provider: str, src_bucket: str, src_region, dst_provider: str, dst_bucket: str, dst_region: str, n_instances: int, n_connections: int):
        self.n_instances = n_instances
        self.n_connections = n_connections
        super().__init__(src_provider, src_region, dst_provider, dst_region)

    def plan(self) -> GatewayProgram:
        src_region_tag = f"{self.src_provider}:{self.src_region}"
        dst_region_tag = f"{self.dst_provider}:{self.dst_region}"
        if src_region_tag == dst_region_tag:  # intra-region transfer w/o solver
            topo = GatewayProgram()
            src_op = GatewayReadObjectStore(self.src_bucket, self.src_region)
            dst_op = GatewayWriteObjectStore(self.dst_bucket, self.dst_region)
            topo.add_operators([src_op, dst_op])
            for _ in range(self.n_instances):
                send_op = GatewaySend(None, dst_region_tag)
                receive_op = GatewayReceive()
                src_op.add_child(send_op)
                send_op.add_child(receive_op)
                
                receive_op.add_child(dst_op)
                topo.add_operators([send_op, receive_op])
            topo.cost_per_gb = 0
            return topo
        else:  # inter-region transfer w/ solver
            topo = GatewayProgram()
            src_op = GatewayReadObjectStore(self.src_bucket, self.src_region)
            dst_op = GatewayWriteObjectStore(self.dst_bucket, self.dst_region)
            topo.add_operators([src_op, dst_op])
            for _ in range(self.n_instances):
                send_op_src = GatewaySend(None, src_region_tag)
                receive_op_src = GatewayReceive()
                send_op_src.add_child(receive_op_src)
                src_op.add_child(send_op_src)

                send_op_dst = GatewaySend(None, dst_region_tag)
                receive_op_dst = GatewayReceive()
                receive_op_src.add_child(send_op_dst)
                send_op_dst.add_child(receive_op_dst)
                receive_op_dst.add_child(dst_op)
                
                topo.add_operators([send_op_src, receive_op_src, send_op_dst, receive_op_dst])
            topo.cost_per_gb = compute.CloudProvider.get_transfer_cost(src_region_tag, dst_region_tag)
            return topo


class ILPSolverPlanner(Planner):
    def __init__(
        self,
        src_provider: str,
        src_region,
        dst_provider: str,
        dst_region: str,
        max_instances: int,
        max_connections: int,
        required_throughput_gbits: float,
    ):
        self.max_instances = max_instances
        self.max_connections = max_connections
        self.solver_required_throughput_gbits = required_throughput_gbits
        super().__init__(src_provider, src_region, dst_provider, dst_region)

    def plan(self) -> ReplicationTopology:
        from skyplane.planner.solver_ilp import ThroughputSolverILP

        problem = ThroughputProblem(
            src=f"{self.src_provider}:{self.src_region}",
            dst=f"{self.dst_provider}:{self.dst_region}",
            required_throughput_gbits=self.solver_required_throughput_gbits,
            gbyte_to_transfer=1,
            instance_limit=self.max_instances,
        )

        with path("skyplane.data", "throughput.csv") as solver_throughput_grid:
            tput = ThroughputSolverILP(solver_throughput_grid)
        solution = tput.solve_min_cost(problem, solver=ThroughputSolverILP.choose_solver(), save_lp_path=None)
        if not solution.is_feasible:
            raise ValueError("ILP solver failed to find a solution, try solving with fewer constraints")
        topo, _ = tput.to_replication_topology(solution)
        return topo


class RONSolverPlanner(Planner):
    def __init__(
        self,
        src_provider: str,
        src_region,
        dst_provider: str,
        dst_region: str,
        max_instances: int,
        max_connections: int,
        required_throughput_gbits: float,
    ):
        self.max_instances = max_instances
        self.max_connections = max_connections
        self.solver_required_throughput_gbits = required_throughput_gbits
        super().__init__(src_provider, src_region, dst_provider, dst_region)

    def plan(self) -> ReplicationTopology:
        from skyplane.planner.solver_ron import ThroughputSolverRON

        problem = ThroughputProblem(
            src=self.src_region,
            dst=self.dst_region,
            required_throughput_gbits=self.solver_required_throughput_gbits,
            gbyte_to_transfer=1,
            instance_limit=self.max_instances,
        )

        with path("skyplane.data", "throughput.csv") as solver_throughput_grid:
            tput = ThroughputSolverRON(solver_throughput_grid)
        solution = tput.solve(problem)
        topo, _ = tput.to_replication_topology(solution)
        return topo
