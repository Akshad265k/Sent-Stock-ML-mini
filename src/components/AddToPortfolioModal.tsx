import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSave: (quantity: number, buyPrice: number) => void;
  currentPrice?: number;
}

const AddToPortfolioModal = ({ isOpen, onClose, onSave, currentPrice }: Props) => {
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");

  // Pre-fill price when modal opens
  useEffect(() => {
    if (isOpen && currentPrice) {
      setPrice(currentPrice.toString());
    }
  }, [isOpen, currentPrice]);

  if (!isOpen) return null;

  const isValid = Number(qty) > 0 && Number(price) > 0;

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={onClose}
          id="portfolio-modal-backdrop"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", duration: 0.4 }}
            onClick={(e) => e.stopPropagation()}
            className="glass-strong rounded-2xl p-6 w-full max-w-sm space-y-5 shadow-2xl"
            id="portfolio-modal"
          >
            <div className="space-y-1">
              <h2 className="text-xl font-semibold text-foreground">
                Add to Portfolio
              </h2>
              {currentPrice && (
                <p className="text-xs text-emerald-400 font-medium mono-num">
                  Live Price: ₹{currentPrice.toLocaleString()}
                </p>
              )}
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  Quantity
                </label>
                <input
                  type="number"
                  placeholder="e.g. 10"
                  className="w-full p-3 mt-1 rounded-xl bg-secondary/50 border border-border text-foreground
                    focus:border-primary/50 focus:ring-1 focus:ring-primary/30 outline-none transition-all"
                  value={qty}
                  onChange={(e) => setQty(e.target.value)}
                  id="modal-quantity-input"
                />
              </div>

              <div>
                <label className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  Buy Price (₹)
                </label>
                <input
                  type="number"
                  placeholder="1500.00"
                  className="w-full p-3 mt-1 rounded-xl bg-secondary/50 border border-border text-foreground
                    focus:border-primary/50 focus:ring-1 focus:ring-primary/30 outline-none transition-all mono-num"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  id="modal-price-input"
                />
              </div>
            </div>

            {/* Total preview */}
            {isValid && (
              <div className="glass rounded-lg p-3 text-center">
                <span className="text-xs text-muted-foreground">Total Investment</span>
                <p className="text-lg font-bold text-foreground mono-num">
                  ₹{(Number(qty) * Number(price)).toLocaleString()}
                </p>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button
                onClick={onClose}
                className="flex-1 py-3 rounded-xl bg-secondary text-muted-foreground hover:bg-secondary/80
                  hover:text-foreground transition-all font-medium"
                id="modal-cancel-btn"
              >
                Cancel
              </button>

              <button
                onClick={() => {
                  if (isValid) {
                    onSave(Number(qty), Number(price));
                    setQty("");
                    setPrice("");
                  }
                }}
                disabled={!isValid}
                className="flex-1 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-400
                  text-black font-bold hover:from-emerald-400 hover:to-emerald-300
                  transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
                id="modal-save-btn"
              >
                Add Now
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AddToPortfolioModal;
