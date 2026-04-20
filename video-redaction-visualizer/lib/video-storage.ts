const DB_NAME = "video-redaction-visualizer";
const DB_VERSION = 1;
const STORE = "uploads";
const CURRENT_KEY = "current";

type StoredVideo = {
  blob: Blob;
  name: string;
  type: string;
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

export async function saveUploadedVideo(file: File): Promise<void> {
  const db = await openDb();
  const payload: StoredVideo = {
    blob: file,
    name: file.name,
    type: file.type || "video/mp4",
  };
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    const put = store.put(payload, CURRENT_KEY);
    put.onerror = () => reject(put.error ?? new Error("save failed"));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("transaction failed"));
  });
}

export async function loadUploadedVideo(): Promise<File | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);
    const get = store.get(CURRENT_KEY);
    get.onerror = () => reject(get.error ?? new Error("load failed"));
    get.onsuccess = () => {
      const raw = get.result as StoredVideo | undefined;
      if (!raw?.blob) {
        resolve(null);
        return;
      }
      resolve(new File([raw.blob], raw.name, { type: raw.type }));
    };
  });
}

export async function clearUploadedVideo(): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    const del = store.delete(CURRENT_KEY);
    del.onerror = () => reject(del.error ?? new Error("clear failed"));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("transaction failed"));
  });
}
