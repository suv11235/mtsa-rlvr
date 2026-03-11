# MTSA RLVR Presentation

This directory contains the slide deck for the Tamper-Resistant Multi-Turn Adversarial RLVR project, built with [React](https://react.dev/), [Vite](https://vitejs.dev/), and [Framer Motion](https://www.framer.com/motion/).

## Prerequisites

- **Node.js**: Version 18 or higher is recommended.
- **npm**: Comes with Node.js.

## Getting Started

Follow these steps to run the presentation on your local machine:

### 1. Install Dependencies

Open your terminal, navigate to this directory, and run:

```bash
npm install
```

This will download all necessary packages specified in `package.json`.

### 2. Run Development Server

To start the presentation locally:

```bash
npm run dev
```

Once the server starts, you will see a URL (usually `http://localhost:5173/`). Open this link in your web browser to view the slides.

## Navigation

- **Next Slide**: Press `Right Arrow` key, `Spacebar`, or click the right chevron button.
- **Previous Slide**: Press `Left Arrow` key or click the left chevron button.

## Building for Production

If you need to create a static version of the presentation (e.g., for hosting or sharing as a zip):

```bash
npm run build
```

This will create a `dist` folder containing the optimized static files.

## Deployment

To deploy this presentation to the web for free using Vercel:

```bash
npx vercel
```
