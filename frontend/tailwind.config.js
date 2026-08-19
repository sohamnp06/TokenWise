/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          darkest: '#08090d',
          dark: '#0e1017',
          surface: '#151824',
          subtle: '#1d2130',
        },
        border: {
          subtle: '#262b3e',
          focus: '#3b4360',
        },
        brand: {
          DEFAULT: '#3b82f6',
          hover: '#2563eb',
          muted: '#1e3a8a',
        }
      }
    },
  },
  plugins: [],
}
