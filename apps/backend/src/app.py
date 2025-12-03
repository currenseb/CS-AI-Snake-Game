import asyncio
import os
import time
import socketio
from aiohttp import web
from typing import Any, Dict, List, Tuple

from agent import DQN
from game import Game

# Create a Socket.IO server instance with CORS allowed for all origins
sio = socketio.AsyncServer(
    cors_allowed_origins="*",  # allow frontend (Next.js) to connect
    async_mode="aiohttp",      # integrate Socket.IO with aiohttp
)

# Create the aiohttp web application
app = web.Application()

# Attach the socket.io server to the aiohttp app
sio.attach(app)

# Basic health check endpoint - keep this for server monitoring
async def handle_ping(request: Any) -> Any:
    """Simple ping endpoint to keep server alive and check if it's running"""
    return web.json_response({"message": "pong"})


@sio.event
async def connect(sid: str, environ: Dict[str, Any]) -> None:
    """
    Handle client connections - called when a frontend connects to the server.

    sid = session id assigned by Socket.IO.
    environ = metadata about the connection (headers, query params, etc.)
    """
    print(f"[connect] Client connected: {sid}")

    # Initialize a new session for this client
    session: Dict[str, Any] = {
        "game": None,   # will hold Game() instance once start_game() is called
        "task": None,   # background asyncio task for update_game()
        # "agent": None  # reserved for later (DQN integration)
    }

    # Save the session for this socket ID
    await sio.save_session(sid, session)

    # Optionally send a welcome message to the frontend
    await sio.emit("server_message", {"msg": "Connected to Snake AI backend"}, room=sid)



@sio.event
async def disconnect(sid: str) -> None:
    """
    Handle client disconnections - cleanup any resources.
    """
    print(f"[disconnect] Client disconnected: {sid}")

    try:
        # Retrieve the session for this client
        session: Dict[str, Any] = await sio.get_session(sid)
    except KeyError:
        # If no session exists (e.g., never saved), just skip
        print(f"[disconnect] No session found for {sid}.")
        return

    # If a background game loop task exists, cancel it
    task = session.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print(f"[disconnect] Canceled background task for {sid}.")

    # Optionally, clear session data
    await sio.save_session(sid, {})

    print(f"[disconnect] Cleanup complete for {sid}.")



@sio.event
async def start_game(sid: str, data: Dict[str, Any]) -> None:
    """
    Start or restart the game in AI or manual (keyboard) mode.
    """
    print(f"[start_game] Start game requested by {sid} with data: {data}")

    # Get or create session
    session: Dict[str, Any] = await sio.get_session(sid)

    # Cancel existing loop if running
    old_task = session.get("task")
    if old_task and not old_task.done():
        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            pass
        print(f"[start_game] Stopped previous loop for {sid}")

    # Create a new game instance
    game = Game()

    # Apply custom tick speed
    if "starting_tick" in data:
        try:
            game.game_tick = float(data["starting_tick"])
        except (ValueError, TypeError):
            print(f"[start_game] Invalid tick value: {data['starting_tick']}")

    # Determine mode
    ai_mode = bool(data.get("ai_mode", False))
    agent = DQN() if ai_mode else None

    # Save new state
    session["game"] = game
    session["agent"] = agent
    await sio.save_session(sid, session)

    # Start new loop task
    loop_task = asyncio.create_task(update_game(sid))
    session["task"] = loop_task
    await sio.save_session(sid, session)

    # Send initial game state to frontend
    await sio.emit("game_state", game.to_dict(), room=sid)

    mode_str = "AI" if ai_mode else "Keyboard"
    print(f"[start_game] Game initialized in {mode_str} mode for {sid}")


@sio.event
async def change_direction(sid: str, data: Dict[str, str]) -> None:
    """
    Handle direction changes from the frontend.
    data = {"direction": "UP" | "DOWN" | "LEFT" | "RIGHT"}
    """
    session = await sio.get_session(sid)
    game = session.get("game")

    if not game or not game.running:
        return

    direction = data.get("direction")
    if direction:
        game.queue_change(direction)


@sio.on("change_speed")
async def change_speed(sid, data):
    """Dynamically adjust the AI snake's tick speed without restarting the game."""
    try:
        session = await sio.get_session(sid)
        game = session.get("game")

        if not game:
            print(f"[change_speed] No game found for {sid}")
            return

        new_tick = float(data.get("new_tick", 0.05))
        old_tick = game.game_tick
        game.game_tick = new_tick

        # Save updated session so it persists between ticks
        await sio.save_session(sid, session)
        print(f"[change_speed] Updated tick for {sid}: {old_tick:.3f} → {new_tick:.3f}")

    except Exception as e:
        print(f"[ERROR] change_speed crashed for {sid}: {e}")


# Event Handlers for Saving/Loading AI Models
@sio.event
async def save_model(sid: str, data: Dict[str, Any]) -> None:
    """
    Save the current AI model for this client's agent.
    """
    print(f"[save_model] Request from {sid}: {data}")

    session = await sio.get_session(sid)
    agent = session.get("agent")

    if agent is None:
        await sio.emit("server_message", {"msg": "No AI agent found to save."}, room=sid)
        return

    try:
        agent.model.save()
        await sio.emit("server_message", {"msg": "Model saved successfully."}, room=sid)
        print(f"[save_model] Model saved for {sid}")
    except Exception as e:
        msg = f"Error saving model: {e}"
        await sio.emit("server_message", {"msg": msg}, room=sid)
        print(f"[save_model] {msg}")


@sio.event
async def load_model(sid: str, data: Dict[str, Any]) -> None:
    """
    Load a previously saved AI model for this client's agent.
    """
    print(f"[load_model] Request from {sid}: {data}")

    session = await sio.get_session(sid)
    agent = session.get("agent")

    if agent is None:
        await sio.emit("server_message", {"msg": "No AI agent to load into."}, room=sid)
        return

    file_name = data.get("file_name")
    if not file_name:
        await sio.emit("server_message", {"msg": "No file name provided for loading."}, room=sid)
        return

    try:
        agent.model.load(file_name)
        await sio.emit("server_message", {"msg": f"Model '{file_name}' loaded successfully."}, room=sid)
        print(f"[load_model] Model '{file_name}' loaded for {sid}")
    except Exception as e:
        msg = f"Error loading model: {e}"
        await sio.emit("server_message", {"msg": msg}, room=sid)
        print(f"[load_model] {msg}")


# Main Game Loop
async def update_game(sid: str) -> None:
    """Main game loop - runs continuously while the game is active."""
    print(f"[update_game] Game loop started for {sid}")

    while True:
        try:
            session: Dict[str, Any] = await sio.get_session(sid)
        except KeyError:
            print(f"[update_game] Session not found for {sid}, stopping loop.")
            break

        game: Game = session.get("game")
        agent: DQN | None = session.get("agent")

        if game is None:
            print(f"[update_game] No game found for {sid}, stopping loop.")
            break

        try:
            if agent is not None:
                # AI mode: agent-driven
                await update_agent_game_state(game, agent)
            else:
                # Manual mode: step automatically every tick
                if game.running:
                    game.step()
                else:
                    # If game isn't running, reset and start it
                    game.reset()
                    game.running = True

            # Send updated state
            await sio.emit("game_state", game.to_dict(), room=sid)
            await sio.save_session(sid, session)

            # Control tick speed
            await asyncio.sleep(game.game_tick)

        except asyncio.CancelledError:
            print(f"[update_game] Game loop cancelled for {sid}")
            break
        except Exception as e:
            print(f"[update_game] Error in loop for {sid}: {e}")
            await asyncio.sleep(0.1)  # prevent tight crash loop

    print(f"[update_game] Loop ended for {sid}")



def action_to_direction(action: List[int], current_dir: Tuple[int, int]) -> str:
    """
    Convert AI action [1,0,0]/[0,1,0]/[0,0,1] into an absolute direction string
    relative to the snake's current movement direction.
    """
    try:
        idx = action.index(1)
    except ValueError:
        idx = 0  # default to "straight" if invalid

    dx, dy = current_dir

    # Map vector to direction
    if (dx, dy) == (0, -1):
        base = "UP"
    elif (dx, dy) == (0, 1):
        base = "DOWN"
    elif (dx, dy) == (-1, 0):
        base = "LEFT"
    else:
        base = "RIGHT"

    # Define right/left turns relative to current direction
    if base == "UP":
        dirs = ["UP", "RIGHT", "LEFT"]
    elif base == "DOWN":
        dirs = ["DOWN", "LEFT", "RIGHT"]
    elif base == "LEFT":
        dirs = ["LEFT", "UP", "DOWN"]
    else:
        dirs = ["RIGHT", "DOWN", "UP"]

    return dirs[idx]



async def update_agent_game_state(game: Game, agent: Any) -> None:
    """
    Handle AI agent decision making and training.

    One full RL step:
      1. Get current state
      2. Choose action (epsilon-greedy)
      3. Apply action to the game
      4. Compute reward
      5. Observe next state
      6. Train short memory
      7. Store experience
      8. If done, train long memory and reset
    """
    try:
        import numpy as np

        state_old = agent.get_state(game)

        action_onehot = agent.get_action(state_old)  # [1,0,0], [0,1,0], or [0,0,1]
        action_idx = int(np.argmax(action_onehot))    # convert to scalar index

        current_dir = game.snake.direction
        direction_str = action_to_direction(action_onehot, current_dir)
        game.queue_change(direction_str)
        game.step()

        done = not game.running

        reward = agent.calculate_reward(game, done)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, action_idx, reward, state_new, done)

        agent.remember(state_old, action_idx, reward, state_new, done)

        if done:
            # Stronger post-game training phase
            for _ in range(10):  # run more long-memory updates per game
                agent.train_long_memory()
            agent.n_games += 1

            # Record score for tracking progress
            if not hasattr(agent, "score_history"):
                agent.score_history = []
            agent.score_history.append(game.score)

            # Compute moving average every 10 games
            if agent.n_games % 10 == 0:
                last_scores = agent.score_history[-10:]
                avg_score = sum(last_scores) / len(last_scores)
                print(
                    f"[AI] Games: {agent.n_games} | "
                    f"Avg (last 10): {avg_score:.2f} | "
                    f"Epsilon: {agent.epsilon:.2f}"
                )

            # Log end of episode
            print(f"[AI] Game {agent.n_games} ended | Score: {game.score}")

            # Reset game for next episode
            game.reset()

            # Optional: quick warmup training to stabilize learning
            for _ in range(3):
                agent.train_long_memory()

            # Reset internal trackers
            agent.prev_distance = None
            agent.prev_score = 0


    except Exception as e:
        print(f"[ERROR] update_agent_game_state crashed: {e}")



async def main() -> None:
    """
    Start the web server and Socket.IO server.
    """
    # Register a simple /ping route for health checks
    app.router.add_get("/ping", handle_ping)

    # Create a web app runner
    # This prepares aiohttp to start serving HTTP + WebSocket traffic
    runner = web.AppRunner(app)
    await runner.setup()

    # Create a site bound to a host and port
    site = web.TCPSite(
        runner,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
    await site.start()

    # Keep the server alive indefinitely
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour and repeat forever
    except KeyboardInterrupt:
        print("\n[main] Keyboard interrupt, shutting down...")
    finally:
        # Gracefully shut down the web app
        await runner.cleanup()



if __name__ == "__main__":
    asyncio.run(main())
