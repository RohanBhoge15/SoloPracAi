/**
 * Drug-name autocomplete for the prescription composer.
 *
 * The list behind it is the FDA NDC directory (~52k brand + generic
 * names, shipped with the backend), served by `GET /api/v1/drugs/suggest`.
 * A keystroke costs the server microseconds, so the only throttle here is
 * a 120 ms debounce to avoid firing on every letter of a fast type.
 *
 * The list renders in a portal, not under the input. `.rx-table` carries
 * `overflow: hidden` (it has clipped things before), so anything
 * absolutely positioned inside the row would be cut off. Fixed positioning
 * from the input's rect sidesteps every clipping ancestor; the list closes
 * on scroll or resize rather than chasing the field.
 */

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "@/lib/api";
import "./composer.css";

const DEBOUNCE_MS = 120;

export default function DrugInput({
  value,
  onChange,
  placeholder,
  ariaLabel,
  inputRef,
}: {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  ariaLabel?: string;
  inputRef?: React.Ref<HTMLInputElement>;
}) {
  const [items, setItems] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const reqId = useRef(0);
  const listId = useId();

  const place = () => {
    const input = wrapRef.current?.querySelector("input");
    if (!input) return false;
    const r = input.getBoundingClientRect();
    // No room below: open upward. A phone keyboard eats the bottom half,
    // so the list must never render where it cannot be seen.
    const below = window.innerHeight - r.bottom;
    const height = Math.min(items.length, 8) * 36 + 8;
    const top = below >= height + 8 ? r.bottom + 4 : Math.max(8, r.top - height - 4);
    setPos({ top, left: r.left, width: Math.max(r.width, 220) });
    return true;
  };

  // Debounced fetch. A request id discards stale replies, so a slow
  // response to "me" can never overwrite the list for "metf".
  useEffect(() => {
    const q = value.trim();
    if (q.length < 2) {
      setItems([]);
      setOpen(false);
      return;
    }
    const id = ++reqId.current;
    const t = window.setTimeout(() => {
      api.suggestDrugs(q).then(
        (r) => {
          if (reqId.current !== id) return;
          setItems(r.suggestions);
          setActive(-1);
          if (r.suggestions.length > 0) {
            setOpen(true);
          } else {
            setOpen(false);
          }
        },
        () => {
          // The list is a convenience, not the field. A failed lookup
          // leaves a working text input, never an error state.
          if (reqId.current !== id) return;
          setItems([]);
          setOpen(false);
        },
      );
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [value]);

  // Position when the list appears; close it when the page moves.
  useEffect(() => {
    if (!open) return;
    place();
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, { capture: true, passive: true });
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, { capture: true });
      window.removeEventListener("resize", close);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, items.length]);

  const pick = (name: string) => {
    reqId.current++;
    setOpen(false);
    setActive(-1);
    onChange(name);
    wrapRef.current?.querySelector("input")?.focus();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open || items.length === 0) {
      if (e.key === "ArrowDown" && items.length > 0) {
        setOpen(true);
        setActive(0);
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      setActive((a) => (a + 1) % items.length);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      setActive((a) => (a - 1 + items.length) % items.length);
      e.preventDefault();
    } else if (e.key === "Enter" && active >= 0) {
      pick(items[active]);
      e.preventDefault();
    } else if (e.key === "Escape") {
      setOpen(false);
      setActive(-1);
    }
  };

  const q = value.trim().toLowerCase();

  return (
    <div className="drugwrap" ref={wrapRef}>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        onBlur={() => {
          // Blur fires before a click on a suggestion. The delay lets the
          // mousedown on the option land first; a plain onMouseDown
          // preventDefault would also stop the input losing focus, which
          // is what keeps the phone keyboard up.
          window.setTimeout(() => setOpen(false), 120);
        }}
        onFocus={() => {
          if (items.length > 0) setOpen(true);
        }}
        placeholder={placeholder}
        aria-label={ariaLabel}
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
        autoComplete="off"
      />
      {open &&
        items.length > 0 &&
        createPortal(
          <ul className="drug-pop" id={listId} role="listbox" style={{ top: pos.top, left: pos.left, width: pos.width }}>
            {items.map((name, i) => {
              const at = name.toLowerCase().indexOf(q);
              return (
                <li
                  key={name}
                  id={`${listId}-${i}`}
                  role="option"
                  aria-selected={i === active}
                  className={i === active ? "on" : ""}
                  onMouseDown={(e) => {
                    // Fires before the input's blur closes the list.
                    e.preventDefault();
                    pick(name);
                  }}
                >
                  {at >= 0 ? (
                    <>
                      {name.slice(0, at)}
                      <b>{name.slice(at, at + q.length)}</b>
                      {name.slice(at + q.length)}
                    </>
                  ) : (
                    name
                  )}
                </li>
              );
            })}
          </ul>,
          document.body,
        )}
    </div>
  );
}
