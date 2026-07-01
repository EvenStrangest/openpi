"""LIBERO-Plus perturbation-robustness eval for openpi policies (pi0.5-LIBERO Phase 2).

This is a fork of ``examples/libero/main.py`` (the stock clean-suite base-gate
eval) retargeted at the LIBERO-Plus perturbation benchmark. The stock loop
enumerates the 10 clean tasks of a suite and runs 50 trials each; here we instead
iterate an explicit task list (one row per *perturbed* LIBERO-Plus task) and run
exactly ONE trial per task, which is the LIBERO-Plus protocol (the perturbation
lives in the task identity, not in repeated trials).

How the perturbation is applied (no code of ours does it -- LIBERO-Plus does):
  * We build the LIBERO-Plus benchmark suite (``libero_spatial`` etc.), which is
    the drop-in-replacement ``libero`` package. Its ``Benchmark`` enumerates all
    ~2.4k perturbed tasks for the suite (task_order_index=0 == identity order).
  * We map each task NAME (column 3 of the task list, identical to the names in
    LIBERO-Plus's task_classification.json / libero_suite_task_map) to its index
    in that suite, then use the suite's OWN resolvers:
      - ``get_task_bddl_file_path(i)``  -> the ENCODED bddl path. Camera-viewpoint,
        robot-initial-state and sensor-noise perturbations are encoded in the
        filename (``..._view_<h>_<v>_<scale>_<r>_<z>_initstate_<N>[_noise_<n>]``)
        and decoded inside ``OffScreenRenderEnv.__init__``; objects-layout /
        background / light perturbations are baked into distinct BDDL files.
      - ``get_task_init_states(i)``     -> the (perturbed-task) init-state array;
        we use index [0] exactly like LIBERO-Plus's own ``render_single_task.py``.
    We pass NO ``robots=`` override, so the default ``["Panda"]`` flows through the
    problem env which prepends the arena prefix (``Mounted``/``OnTheGround``) and,
    for robot-init tasks, appends the variant index -> a registered robot class.
  * Goal predicates are preserved across all perturbations, so the environment's
    ``done`` flag remains a valid success signal (same as the clean suites).

Everything else -- the 180-degree image rotation, resize-with-pad to 224, the
state vector (eef pos + axis-angle + gripper qpos), the replan cadence, the
websocket client contract -- is copied VERBATIM from the stock main.py so that
pi0.5 sees byte-identical observations to the base-gate run. The only behavioural
differences are (a) task enumeration and (b) 1 trial/task.

Outputs, per task: one JSON line appended to ``results_jsonl`` (resumable: a task
whose name already appears there is skipped) and one agentview replay mp4.

Written 2026-07-01 for the pi0.5-LIBERO LIBERO-Plus geometric-axes eval on S03.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Per-suite step budgets, identical to the stock base-gate main.py.
MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO-Plus environment-specific parameters
    #################################################################################################################
    # The LIBERO-Plus benchmark suite whose perturbed tasks we index into. Must be
    # the suite that the task list belongs to (libero_spatial / libero_object /
    # libero_goal / libero_10).
    task_suite_name: str = "libero_spatial"
    # Path (inside the container) to a TSV of perturbed tasks to run, one per line:
    #   <category>\t<level>\t<task_name>
    task_list: str = ""
    # Optional comma-separated category allow-list (empty == run every row in the TSV).
    categories: str = ""
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    max_steps: int = 0  # 0 == use the per-suite default from MAX_STEPS

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"  # Path to save replay videos
    results_jsonl: str = "data/libero_plus/results.jsonl"  # Per-task results (append; resumable)
    record_video: bool = True
    seed: int = 7  # Random Seed (for reproducibility)


def _read_task_list(path, category_filter):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            category, level, task_name = parts[0], parts[1], parts[2]
            if category_filter and category not in category_filter:
                continue
            rows.append((category, level, task_name))
    return rows


def _load_done_tasks(results_jsonl):
    done = set()
    p = pathlib.Path(results_jsonl)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task" in rec:
                done.add(rec["task"])
    return done


def eval_libero_plus(args: Args) -> None:
    np.random.seed(args.seed)
    logging.info(f"Task suite: {args.task_suite_name}")

    if not args.task_list:
        raise ValueError("--args.task-list is required (path to a <category>\\t<level>\\t<task> TSV).")

    max_steps = args.max_steps if args.max_steps > 0 else MAX_STEPS[args.task_suite_name]

    category_filter = set(c.strip() for c in args.categories.split(",") if c.strip())
    rows = _read_task_list(args.task_list, category_filter)
    logging.info(f"Loaded {len(rows)} task rows from {args.task_list}"
                 + (f" (categories={sorted(category_filter)})" if category_filter else ""))

    # Build the LIBERO-Plus benchmark suite and a name->index map (identity order).
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    name_to_id = {name: i for i, name in enumerate(task_suite.get_task_names())}
    logging.info(f"Benchmark suite {args.task_suite_name}: {task_suite.n_tasks} perturbed tasks registered")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)

    done_tasks = _load_done_tasks(args.results_jsonl)
    if done_tasks:
        logging.info(f"Resume: {len(done_tasks)} tasks already have results; will skip them")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for category, level, task_name in tqdm.tqdm(rows):
        if task_name in done_tasks:
            continue

        rec = {
            "suite": args.task_suite_name,
            "category": category,
            "level": level,
            "task": task_name,
        }

        if task_name not in name_to_id:
            logging.error(f"Task '{task_name}' not found in suite {args.task_suite_name}; recording as error")
            rec.update({"success": False, "steps": 0, "error": "task_name_not_in_suite"})
            _append_result(args.results_jsonl, rec)
            continue

        task_id = name_to_id[task_name]
        task = task_suite.get_task(task_id)

        env = None
        done = False
        t_used = 0
        try:
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task_suite, task_id, task, LIBERO_ENV_RESOLUTION, args.seed)

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[0])

            t = 0
            replay_images = []
            while t < max_steps + args.num_steps_wait:
                # Wait for objects to settle before acting (they drop on reset).
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, "
                        f"but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                t += 1
                if done:
                    break

            t_used = t
            if done:
                total_successes += 1

            if args.record_video and replay_images:
                _write_video(args.video_out_path, category, level, task_name, done, replay_images)

        except Exception as e:  # noqa: BLE001 -- one bad task must not kill the shard
            logging.error(f"Caught exception on task '{task_name}': {e}")
            rec["error"] = repr(e)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

        total_episodes += 1
        rec.update({"success": bool(done), "steps": int(t_used)})
        _append_result(args.results_jsonl, rec)

        logging.info(
            f"[{category}/L{level}] {task_name} -> success={done} "
            f"(running {total_successes}/{total_episodes} = "
            f"{100.0 * total_successes / max(total_episodes, 1):.1f}%)"
        )

    if total_episodes:
        logging.info(
            f"Shard done. Success rate: {total_successes / total_episodes:.4f} "
            f"({total_successes}/{total_episodes})"
        )
    else:
        logging.info("Shard done. No new tasks were run (all already had results).")


def _append_result(results_jsonl, rec):
    with open(results_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def _write_video(video_out_path, category, level, task_name, success, frames):
    suffix = "success" if success else "failure"
    cat_slug = category.replace(" ", "-")
    # Bound the filename length (task names are long); keep it unambiguous.
    stem = f"{cat_slug}__L{level}__{task_name}__{suffix}"[:200]
    imageio.mimwrite(
        pathlib.Path(video_out_path) / f"{stem}.mp4",
        [np.asarray(x) for x in frames],
        fps=10,
    )


def _get_libero_env(task_suite, task_id, task, resolution, seed):
    """Build the LIBERO-Plus env for a perturbed task, mirroring render_single_task.py.

    We use the benchmark's own ``get_task_bddl_file_path`` so that the encoded
    camera/robot/noise perturbation string reaches OffScreenRenderEnv intact.
    """
    task_description = task.language
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed affects object positions even with a fixed init state.
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite (transform_utils.quat2axisangle)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_plus)
