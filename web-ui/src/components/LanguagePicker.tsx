/**
 * Query-language selector: the 14 indexed Indic languages plus auto-detect.
 *
 * Auto-detect sends `lang: null`, which is what the backend keys its script detection off.
 * Every option shows its own endonym in its own script -- which doubles as a live check
 * that the font fallback chain resolved, because a missing Noto face shows up here as a
 * row of tofu boxes before it ever reaches an answer.
 */

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { LANGS, LANG_ENDONYM, LANG_LABEL, type Lang } from '@/lib/types'
import { cn, langAttrs } from '@/lib/utils'

/** Radix forbids an empty option value, so auto-detect needs a sentinel of its own. */
const AUTO = 'auto'

export interface LanguagePickerProps {
  /** null means auto-detect. */
  value: Lang | null
  onChange: (lang: Lang | null) => void
  disabled?: boolean
  className?: string
  id?: string
}

export function LanguagePicker({ value, onChange, disabled, className, id }: LanguagePickerProps) {
  return (
    <Select
      value={value ?? AUTO}
      onValueChange={(next) => onChange(next === AUTO ? null : (next as Lang))}
      disabled={disabled}
    >
      <SelectTrigger
        id={id}
        size="default"
        aria-label="Query language"
        className={cn('h-9 min-w-44', className)}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent position="popper" align="start" className="max-h-80">
        <SelectGroup>
          <SelectItem value={AUTO} textValue="Auto-detect">
            <span className="font-medium">Auto-detect</span>
            <span className="text-xs text-muted-foreground">from script</span>
          </SelectItem>
        </SelectGroup>
        <SelectSeparator />
        <SelectGroup>
          {LANGS.map((lang) => (
            // `textValue` keeps Radix's type-ahead usable: typing "ta" cannot match தமிழ்.
            <SelectItem key={lang} value={lang} textValue={`${LANG_LABEL[lang]} ${lang}`}>
              {/* dir sits on the endonym alone -- putting it on the row would flip the
                  Urdu option's layout instead of just its text. */}
              <span className="leading-[1.6]" {...langAttrs(lang)}>
                {LANG_ENDONYM[lang]}
              </span>
              <span className="text-xs text-muted-foreground">{LANG_LABEL[lang]}</span>
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}
