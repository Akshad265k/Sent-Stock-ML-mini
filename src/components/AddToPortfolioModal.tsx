import React, { useState } from "react";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSave: (quantity: number, buyPrice: number) => void;
}

const AddToPortfolioModal = ({ isOpen, onClose, onSave }: Props) => {
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState("");

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-gray-900 p-6 rounded-2xl border border-gray-700 w-full max-w-sm space-y-5">
        <h2 className="text-xl font-semibold text-gray-100">
          Add to Portfolio
        </h2>

        <div className="space-y-3">
          <div>
            <label className="text-gray-400 text-sm">Quantity</label>
            <input
              type="number"
              className="w-full p-2 rounded-md bg-gray-800 border border-gray-600 text-gray-100"
              value={qty}
              onChange={(e) => setQty(e.target.value)}
            />
          </div>

          <div>
            <label className="text-gray-400 text-sm">Buy Price</label>
            <input
              type="number"
              className="w-full p-2 rounded-md bg-gray-800 border border-gray-600 text-gray-100"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-md bg-gray-700 text-gray-300 hover:bg-gray-600"
          >
            Cancel
          </button>

          <button
            onClick={() => {
              onSave(Number(qty), Number(price));
              setQty("");
              setPrice("");
            }}
            className="flex-1 py-2 rounded-md bg-emerald-600 text-black font-semibold hover:bg-emerald-500"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddToPortfolioModal;
