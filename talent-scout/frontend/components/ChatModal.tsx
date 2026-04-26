"use client";

import { ConversationMessage } from "@/app/types";

interface ChatModalProps {
  name: string;
  messages: ConversationMessage[];
  onClose: () => void;
}

export default function ChatModal({ name, messages, onClose }: ChatModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="card w-full max-w-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Recruiter conversation with <span className="text-sky-400">{name}</span>
          </h3>
          <button
            onClick={onClose}
            className="rounded-lg px-3 py-1 text-sm text-slate-400 hover:bg-slate-700 hover:text-white"
          >
            ✕ Close
          </button>
        </div>

        <div className="flex max-h-[60vh] flex-col gap-3 overflow-y-auto pr-1">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "recruiter" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm leading-relaxed ${
                  msg.role === "recruiter"
                    ? "bg-sky-600 text-white"
                    : "bg-slate-700 text-slate-100"
                }`}
              >
                <p className="mb-1 text-xs font-semibold opacity-60 capitalize">{msg.role}</p>
                {msg.content}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
