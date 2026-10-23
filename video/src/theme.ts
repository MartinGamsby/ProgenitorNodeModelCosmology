import {loadFont as loadMontserrat} from '@remotion/google-fonts/Montserrat';
import {loadFont as loadInter} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

export {FPS, WIDTH, HEIGHT, T, VERTICAL} from './constants';

const montserrat = loadMontserrat('normal', {weights: ['600', '800', '900'], subsets: ['latin']});
const inter = loadInter('normal', {weights: ['400', '600', '700'], subsets: ['latin']});
const mono = loadMono('normal', {weights: ['500'], subsets: ['latin']});

export const FONT = {
  display: montserrat.fontFamily,
  body: inter.fontFamily,
  mono: mono.fontFamily,
};

/** Colour convention matches the paper's figures (hubble_diagram.py). */
export const C = {
  bg: '#04060d',
  bg2: '#0a1022',
  ink: '#eef2ff',
  dim: '#8e9bc0',
  faint: '#3a4566',
  lcdm: '#4ea8ff', // ΛCDM: blue
  model: '#ff9a3c', // External-Node / best observer: orange
  eds: '#5ee08a', // EdS / no dark energy: green
  node: '#b98cff', // HMEAs
  nodeGlow: '#7d4dff',
  hot: '#ff5470', // warnings, knots
  gold: '#ffd84a', // highlights, WHAT IF
  bubble: '#5fd0ff', // our observable universe
};
