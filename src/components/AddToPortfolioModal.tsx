import React, { useState, useEffect } from "react";

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

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 p-6 rounded-2xl border border-gray-700 w-full max-w-sm space-y-5 shadow-2xl">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold text-gray-100">
            Add to Portfolio
          </h2>
          {currentPrice && (
            <p className="text-xs text-emerald-400 font-medium">
              Live Price: ₹{currentPrice.toLocaleString()}
            </p>
          )}
        </div>

        <div className="space-y-3">
          <div>
            <label className="text-gray-400 text-xs font-medium uppercase tracking-wider">Quantity</label>
            <input
              type="number"
              placeholder="e.g. 10"
              className="w-full p-3 mt-1 rounded-xl bg-gray-800 border border-gray-700 text-gray-100 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all"
              value={qty}
              onChange={(e) => setQty(e.target.value)}
            />
          </div>

          <div>
            <label className="text-gray-400 text-xs font-medium uppercase tracking-wider">Buy Price (₹)</label>
            <input
              type="number"
              placeholder="1500.00"
              className="w-full p-3 mt-1 rounded-xl bg-gray-800 border border-gray-700 text-gray-100 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition-all"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            className="flex-1 py-3 rounded-xl bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-100 transition-colors font-medium"
          >
            Cancel
          </button>

          <button
            onClick={() => {
              onSave(Number(qty), Number(price));
              setQty("");
              setPrice("");
            }}
            className="flex-1 py-3 rounded-xl bg-emerald-500 text-black font-bold hover:bg-emerald-400 transition-colors shadow-lg shadow-emerald-500/20"
          >
            Add Now
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddToPortfolioModal;
