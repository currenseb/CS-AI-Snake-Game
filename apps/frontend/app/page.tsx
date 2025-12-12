"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";

interface GameState {
  grid_width: number;
  grid_height: number;
  game_tick: number;
  snake: [number, number][];
  food: [number, number];
  score: number;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<Socket | null>(null);

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number } | null>(null);
  const [aiMode, setAiMode] = useState(true);
  const [aiSpeed, setAiSpeed] = useState(0.02);
  const [showPopup, setShowPopup] = useState(false);
  const keyboardSpeed = 0.1;

  // --- Initialize socket once ---
  useEffect(() => {
    if (socketRef.current) return;

    const socket = io(process.env.NEXT_PUBLIC_BACKEND_URL!, {
      transports: ["websocket"],
      reconnection: true,
      path: "/socket.io/",
    });

    socketRef.current = socket;

    socket.on("connect", () => {
      console.log("[frontend] connected to backend");
      socket.emit("start_game", {
        starting_tick: aiMode ? aiSpeed : keyboardSpeed,
        ai_mode: aiMode,
      });
    });

    socket.on("disconnect", () => {
      console.warn("[frontend] disconnected from backend");
    });

    socket.on("game_state", (data: GameState) => {
      setGameState(data);
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  // --- Mode toggle ---
  useEffect(() => {
    const socket = socketRef.current;
    if (socket && socket.connected) {
      console.log(`[frontend] switching mode -> ${aiMode ? "AI" : "Keyboard"}`);
      setGameState(null);
      socket.emit("start_game", {
        starting_tick: aiMode ? aiSpeed : keyboardSpeed,
        ai_mode: aiMode,
      });
    }
  }, [aiMode]);

  // --- Live AI speed updates ---
  useEffect(() => {
    if (aiMode && socketRef.current && socketRef.current.connected) {
      socketRef.current.emit("change_speed", { new_tick: aiSpeed });
    }
  }, [aiSpeed, aiMode]);

  // --- Keyboard input ---
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!socketRef.current || aiMode) return;

      let direction: string | null = null;
      switch (event.key) {
        case "ArrowUp":
          direction = "UP";
          break;
        case "ArrowDown":
          direction = "DOWN";
          break;
        case "ArrowLeft":
          direction = "LEFT";
          break;
        case "ArrowRight":
          direction = "RIGHT";
          break;
      }

      if (direction) {
        socketRef.current.emit("change_direction", { direction });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [aiMode]);

  // --- Draw game ---
  const drawGame = (
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    state: GameState
  ) => {
    const { grid_width, grid_height, snake, food } = state;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellSize = Math.floor(
      Math.min(canvas.width / grid_width, canvas.height / grid_height)
    );
    const offsetX = (canvas.width - grid_width * cellSize) / 2;
    const offsetY = (canvas.height - grid_height * cellSize) / 2;

    for (let x = 0; x < grid_width; x++) {
      for (let y = 0; y < grid_height; y++) {
        ctx.fillStyle = (x + y) % 2 === 0 ? "#101010" : "#0d0d0d";
        ctx.fillRect(
          offsetX + x * cellSize,
          offsetY + y * cellSize,
          cellSize,
          cellSize
        );
      }
    }

    snake.forEach(([x, y], idx) => {
      const px = offsetX + x * cellSize;
      const py = offsetY + y * cellSize;
      ctx.fillStyle = idx === 0 ? "#4ade80" : "#22c55e";
      ctx.fillRect(px, py, cellSize, cellSize);
    });

    const [fx, fy] = food;
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(
      offsetX + fx * cellSize,
      offsetY + fy * cellSize,
      cellSize,
      cellSize
    );

    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    for (let i = 0; i <= grid_width; i++) {
      ctx.beginPath();
      ctx.moveTo(offsetX + i * cellSize, offsetY);
      ctx.lineTo(offsetX + i * cellSize, offsetY + grid_height * cellSize);
      ctx.stroke();
    }
    for (let j = 0; j <= grid_height; j++) {
      ctx.beginPath();
      ctx.moveTo(offsetX, offsetY + j * cellSize);
      ctx.lineTo(offsetX + grid_width * cellSize, offsetY + j * cellSize);
      ctx.stroke();
    }
  };

  // ✅ THE ONE ADDED EFFECT (this was missing)
  useEffect(() => {
    if (!canvasSize) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (gameState) {
      drawGame(ctx, canvas, gameState);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [gameState, canvasSize]);

  // --- Resize handling ---
  useEffect(() => {
    const computeSize = () => {
      const maxBoard = Math.min(
        window.innerWidth * 0.9,
        window.innerHeight * 0.7
      );

      const width = Math.max(900, maxBoard);
      const height = Math.max(506.25, maxBoard);

      setCanvasSize({ width: width, height: height });
    };

    computeSize();
    window.addEventListener("resize", computeSize);
    return () => window.removeEventListener("resize", computeSize);
  }, []);

  return (
    <div className="w-full flex flex-col items-center justify-start pt-6 gap-4 px-4">

      {/* Controls row */}
      <div className="flex flex-wrap items-center justify-center gap-4 text-center">
        <button
          onClick={() => setAiMode((prev) => !prev)}
          className={`px-4 py-2 rounded-md font-semibold transition ${
            aiMode
              ? "bg-green-600 hover:bg-green-700 text-white"
              : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
        >
          {aiMode ? "Play Yourself!" : "Watch The AI Snake Learn!"}
        </button>

        {aiMode && (
          <button
            onClick={() => setShowPopup(true)}
            className="px-4 py-2 rounded-md bg-yellow-600 hover:bg-yellow-700 text-white font-semibold transition"
          >
            Why Does It Learn So Slowly?
          </button>
        )}

        <span className="text-gray-300 text-sm">
          {gameState ? `Score: ${gameState.score}` : "Connecting..."}
        </span>

        <span className="text-gray-400 text-xs">
          Mode: {aiMode ? "AI" : "Keyboard"}
        </span>
      </div>

      {aiMode && (
        <div className="flex flex-col items-center mt-2">
          <label htmlFor="speed-slider" className="text-sm text-gray-300 mb-1">
            <span>Snake Speed</span>
          </label>
          <div className="flex items-center gap-2">
            <span className="text-gray-500 text-xs">🐢 Slow</span>
            <input
              id="speed-slider"
              type="range"
              min="0.01"
              max="0.2"
              step="0.005"
              value={0.21 - aiSpeed}
              onChange={(e) =>
                setAiSpeed(0.21 - parseFloat(e.target.value))
              }
              className="w-64 accent-green-500"
            />
            <span className="text-gray-500 text-xs">⚡ Fast</span>
          </div>
        </div>
      )}

      {canvasSize && (
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          className="shadow-lg"
          style={{
            border: "none",
            outline: "none",
            display: "block",
            backgroundColor: "black",
            filter: showPopup ? "blur(5px)" : "none",
            transition: "filter 0.2s ease",
          }}
        />
      )}

      {showPopup && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/10 bg-opacity-50 backdrop-blur-md z-50">
          <div className="bg-gray-900/80 text-white rounded-lg p-6 max-w-md shadow-lg relative text-center">
            <button
              onClick={() => setShowPopup(false)}
              className="absolute top-2 right-3 text-gray-400 hover:text-white"
            >
              ✕
            </button>
            <h2 className="text-2xl font-bold mb-3 text-green-400">
              Why Does It Learn So Slowly?
            </h2>
            <p className="text-md text-gray-300 leading-relaxed">
              When the AI plays, it isn’t following a prewritten strategy,
              it’s learning by trial and error. Each move gives it a little bit of feedback:
              rewards when it moves toward food, penalties when it crashes. Over many thousands
              of moves, it gradually discovers patterns that lead to higher scores. If it learned
              too fast or randomly changed its behavior after every success, it would just memorize
              lucky outcomes instead of understanding what truly works. The slower pace lets it build
              consistent skill, just like a person learning a new game by playing carefully and
              reflecting after mistakes. So while it might look hesitant early on, that’s intentional,
              the AI is gathering experiences and slowly reinforcing behaviors that actually lead to
              long-term success rather than quick, lucky wins. The snake is learning is a similar
              fashion to that of a human, except the AI-powered snake never gets tired, and can
              keep doing this forever, while a human will get tired or quit out of frustration.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

