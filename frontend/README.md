# MedQuery AI: frontend

React + TypeScript web app for MedQuery AI, built with [Vite](https://vite.dev/).

## Requirements

Node.js 20.19+ (developed with Node 24) and npm.

## Commands

Run these from this `frontend/` folder.

| Command | What it does |
|---|---|
| `npm install` | Download dependencies into `node_modules/` (first time only) |
| `npm run dev` | Start the development server at http://localhost:5173 |
| `npm test` | Run the automated tests once |
| `npm run lint` | Check the code for common mistakes |
| `npm run build` | Type-check and build the production version into `dist/` |

On Windows PowerShell, if you see "running scripts is disabled on this system",
use `npm.cmd` instead of `npm` (for example `npm.cmd run dev`).

## Notes

- TypeScript is pinned to 6.0 because `typescript-eslint` does not yet support
  TypeScript 7.
