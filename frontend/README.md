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

## Pages

| Page | Address | Shows |
|---|---|---|
| Dashboard | `/` | Dataset facts, the backend connection and the selected model |
| Chat | `/chat` | Placeholder; grounded answers arrive in Part 3 |
| Retrieval | `/retrieval` | Ask a question and see the chunks a model retrieves, with similarity scores (needs the backend) |
| Knowledge Base | `/knowledge-base` | The 698 research articles and their metadata |
| Model Lab | `/model-lab` | Choose one of the four embedding models |
| Analytics | `/analytics` | The research's historical retrieval results |

The Retrieval page calls the backend's `POST /api/retrieve` through the development server
(see `../backend/README.md`), so start both. Its search is kept in the address
(`/retrieval?q=...&model=BGE&k=5`), so Back, refresh and shared links repeat a search.

## Notes

- TypeScript is pinned to 6.0 because `typescript-eslint` does not yet support
  TypeScript 7.
