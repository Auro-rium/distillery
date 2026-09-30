/**
 * A long identifier (a counter or purpose name from the API) that may wrap after an underscore, and
 * only inside a word as a last resort. Each part is an inline-block, which gives the line breaker a
 * break opportunity between parts; the text content is exactly the input, with no inserted characters
 * (an inserted `<wbr>` shows up as a space in some accessible-name computations).
 */
export function Breakable({ text }: { text: string }) {
  return (
    <>
      {text.split(/(?<=_)/).map((p, i) => <span key={i} className="brk">{p}</span>)}
    </>
  );
}
