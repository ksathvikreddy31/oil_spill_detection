import { render, screen } from '@testing-library/react';
import App from './App';

test('renders main heading', () => {
  render(<App />);
  const headingElement = screen.getByText(/Hybrid Oil Spill Detection System/i);
  expect(headingElement).toBeInTheDocument();
});
